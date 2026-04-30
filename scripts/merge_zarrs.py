#!/usr/bin/env python3
"""
Merge two or more UMI replay-buffer zarrs into one that looks exactly like
a single recording session.

The output zarr will have:
  - All episodes from every input, in the order listed
  - meta/episode_ends that is cumulative and always equals total frames
  - The same chunks and compressors as the first input zarr
  - Every data key that appears in the first input must also appear in all others

Usage:
  python scripts/merge_zarrs.py \\
      data/session_a/replay_buffer.zarr \\
      data/session_b/replay_buffer.zarr \\
      --output data/merged/replay_buffer.zarr
"""

import sys
import pathlib
import argparse
import numpy as np
import zarr
import imagecodecs.numcodecs
imagecodecs.numcodecs.register_codecs()

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)

from diffusion_policy.common.replay_buffer import ReplayBuffer


def open_source(path: str) -> ReplayBuffer:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {path}")
    if p.is_file():
        # ZipStore (.zarr files that are actually zip archives)
        store = zarr.ZipStore(str(p), mode='r')
        root = zarr.open(store, mode='r')
        return ReplayBuffer(root)
    else:
        return ReplayBuffer.create_from_path(str(p), mode='r')


def validate_all(buffers: list, paths: list):
    """
    Every zarr must have the same keys, dtypes, and non-time array shapes.
    Raises ValueError with a clear message if anything mismatches.
    """
    ref = buffers[0]
    ref_keys = sorted(ref.keys())

    for i, (buf, path) in enumerate(zip(buffers[1:], paths[1:]), start=1):
        keys = sorted(buf.keys())
        if keys != ref_keys:
            missing = set(ref_keys) - set(keys)
            extra   = set(keys) - set(ref_keys)
            raise ValueError(
                f"Key mismatch in zarr #{i} ({path}).\n"
                f"  Missing from it : {sorted(missing)}\n"
                f"  Extra in it     : {sorted(extra)}"
            )
        for key in ref_keys:
            r_arr = ref.data[key]
            t_arr = buf.data[key]
            if r_arr.dtype != t_arr.dtype:
                raise ValueError(
                    f"Dtype mismatch for '{key}': "
                    f"zarr #0 has {r_arr.dtype}, zarr #{i} has {t_arr.dtype}"
                )
            if r_arr.shape[1:] != t_arr.shape[1:]:
                raise ValueError(
                    f"Shape mismatch for '{key}': "
                    f"zarr #0 has per-frame shape {r_arr.shape[1:]}, "
                    f"zarr #{i} has {t_arr.shape[1:]}"
                )


def merge(input_paths: list, output_path: str):
    # ── open all sources ────────────────────────────────────────────────────
    print(f"Opening {len(input_paths)} source zarr(s):")
    buffers = []
    for p in input_paths:
        buf = open_source(p)
        buffers.append(buf)
        print(f"  {p}")
        print(f"    episodes : {buf.n_episodes}")
        print(f"    frames   : {buf.n_steps}")
        print(f"    keys     : {sorted(buf.keys())}")

    # ── compatibility check ─────────────────────────────────────────────────
    print("\nValidating compatibility across all inputs...")
    validate_all(buffers, input_paths)
    print("  OK")

    # ── capture chunk/compressor layout from first zarr ────────────────────
    ref = buffers[0]
    chunks      = ref.get_chunks()      if ref.backend == 'zarr' else {}
    compressors = ref.get_compressors() if ref.backend == 'zarr' else {}

    # ── create output ───────────────────────────────────────────────────────
    out_path = pathlib.Path(output_path)
    if out_path.exists():
        raise FileExistsError(
            f"Output already exists: {output_path}\n"
            "Delete it first or pick a different path."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build in memory first, then write to ZipStore so the output is a .zarr
    # zip file compatible with UmiDataset (which always opens via zarr.ZipStore).
    mem_store = zarr.MemoryStore()
    out       = ReplayBuffer.create_empty_zarr(storage=mem_store)

    # ── copy episodes one by one ────────────────────────────────────────────
    total_eps = sum(b.n_episodes for b in buffers)
    done      = 0

    print(f"\nMerging {total_eps} total episodes → {output_path}")
    for src_path, src_buf in zip(input_paths, buffers):
        for ep_idx in range(src_buf.n_episodes):
            ep = src_buf.get_episode(ep_idx)
            # zarr slices return zarr/numpy arrays; ensure plain numpy for add_episode
            ep_np = {k: np.asarray(v) for k, v in ep.items()}
            out.add_episode(
                ep_np,
                chunks=chunks,
                compressors=compressors,
            )
            done += 1
            print(f"\r  {done}/{total_eps}", end='', flush=True)
        print(f"\r  {done}/{total_eps}  ✓  {src_path}")

    # ── integrity check ─────────────────────────────────────────────────────
    print("\nRunning integrity check...")

    assert out.n_episodes == total_eps, (
        f"Episode count wrong: expected {total_eps}, got {out.n_episodes}"
    )
    assert int(out.episode_ends[-1]) == out.n_steps, (
        f"episode_ends[-1]={out.episode_ends[-1]} != n_steps={out.n_steps}"
    )
    for key in out.keys():
        arr_len = out.data[key].shape[0]
        assert arr_len == out.n_steps, (
            f"Array '{key}' has {arr_len} frames but episode_ends says {out.n_steps}"
        )

    # verify cumulative ordering is strictly increasing
    ends = out.episode_ends[:]
    assert np.all(np.diff(ends) > 0), "episode_ends is not strictly increasing"

    ep_lens = out.episode_lengths
    print("  Passed.")

    # ── write memory store → ZipStore ───────────────────────────────────────
    print(f"\nWriting ZipStore → {output_path} ...")
    with zarr.ZipStore(str(out_path), mode='w') as zip_store:
        zarr.copy_store(source=mem_store, dest=zip_store)
    print("  Done.")

    print(f"\nResult:")
    print(f"  Episodes : {out.n_episodes}")
    print(f"  Frames   : {out.n_steps}")
    print(f"  Ep len   : min={ep_lens.min()}  max={ep_lens.max()}  mean={ep_lens.mean():.1f}")
    print(f"  Keys     : {sorted(out.keys())}")
    print(f"  Output   : {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge UMI replay-buffer zarrs into one seamless zarr",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('inputs', nargs='+', metavar='INPUT_ZARR',
                        help="Two or more input zarr paths, merged in this order")
    parser.add_argument('--output', '-o', required=True, metavar='OUTPUT_ZARR',
                        help="Path to write the merged zarr (must not exist)")
    args = parser.parse_args()

    if len(args.inputs) < 2:
        parser.error("Provide at least 2 input zarrs.")

    merge(args.inputs, args.output)


if __name__ == '__main__':
    main()
