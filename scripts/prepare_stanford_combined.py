#!/usr/bin/env python3
"""
Patch Stanford cup_in_the_lab.zarr with robot0_demo_start_pose, then merge
with a user zarr to create a combined training dataset.

Stanford zarr is missing robot0_demo_start_pose (needed by UmiDataset to
compute robot0_eef_rot_axis_angle_wrt_start). We compute it from the first
frame of each episode: concat(eef_pos[0], eef_rot_axis_angle[0]).

Also strips robot0_demo_end_pose from user zarr (not used by UmiDataset,
present in user data but absent from Stanford — keeps keys compatible).

Output zarr contains: camera0_rgb, robot0_demo_start_pose, robot0_eef_pos,
robot0_eef_rot_axis_angle, robot0_gripper_width.

Usage:
  python scripts/prepare_stanford_combined.py \\
      --stanford zarrs/cup_in_the_lab.zarr \\
      --user     zarrs/0430_replay_buffer.zarr \\
      --output   zarrs/stanford_0430_combined.zarr
"""

import argparse
import pathlib
import sys
import numpy as np
import zarr
import imagecodecs.numcodecs
imagecodecs.numcodecs.register_codecs()

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)

from diffusion_policy.common.replay_buffer import ReplayBuffer

KEEP_KEYS = {
    'camera0_rgb',
    'robot0_demo_start_pose',
    'robot0_eef_pos',
    'robot0_eef_rot_axis_angle',
    'robot0_gripper_width',
}


def open_buf(path: str) -> ReplayBuffer:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {path}")
    if p.is_file():
        store = zarr.ZipStore(str(p), mode='r')
        return ReplayBuffer(zarr.open(store, mode='r'))
    return ReplayBuffer.create_from_path(str(p), mode='r')


def patch_demo_start_pose(ep: dict) -> dict:
    """Broadcast first-frame pose to every step — matches how user data stores it."""
    n = ep['robot0_eef_pos'].shape[0]
    start = np.concatenate([
        ep['robot0_eef_pos'][0],
        ep['robot0_eef_rot_axis_angle'][0],
    ])  # (6,)
    ep['robot0_demo_start_pose'] = np.tile(start, (n, 1)).astype(np.float32)
    return ep


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stanford', required=True, metavar='PATH',
                        help='Stanford cup_in_the_lab.zarr (directory or zip)')
    parser.add_argument('--user', required=True, metavar='PATH',
                        help='User zarr (e.g. 0430_replay_buffer.zarr)')
    parser.add_argument('--output', '-o', required=True, metavar='PATH',
                        help='Output zarr path (must not exist)')
    args = parser.parse_args()

    out_path = pathlib.Path(args.output)
    if out_path.exists():
        raise FileExistsError(f"Output already exists: {args.output}\nDelete it first.")

    print(f'Loading Stanford zarr: {args.stanford}')
    stanford_buf = open_buf(args.stanford)
    print(f'  {stanford_buf.n_episodes} episodes  {stanford_buf.n_steps} steps')
    print(f'  keys: {sorted(stanford_buf.keys())}')

    print(f'\nLoading user zarr: {args.user}')
    user_buf = open_buf(args.user)
    print(f'  {user_buf.n_episodes} episodes  {user_buf.n_steps} steps')
    print(f'  keys: {sorted(user_buf.keys())}')

    total_eps = stanford_buf.n_episodes + user_buf.n_episodes
    mem_store = zarr.MemoryStore()
    out = ReplayBuffer.create_empty_zarr(storage=mem_store)
    done = 0

    print(f'\nMerging {total_eps} total episodes...')

    # Stanford: inject demo_start_pose, filter to KEEP_KEYS
    for i in range(stanford_buf.n_episodes):
        ep = {k: np.asarray(v) for k, v in stanford_buf.get_episode(i).items()}
        ep = patch_demo_start_pose(ep)
        ep = {k: v for k, v in ep.items() if k in KEEP_KEYS}
        out.add_episode(ep)
        done += 1
        print(f'\r  [{done}/{total_eps}] Stanford ep {i}', end='', flush=True)
    print(f'\r  [{done}/{total_eps}] Stanford done      ')

    # User: drop demo_end_pose (absent from Stanford), filter to KEEP_KEYS
    for i in range(user_buf.n_episodes):
        ep = {k: np.asarray(v) for k, v in user_buf.get_episode(i).items()}
        ep = {k: v for k, v in ep.items() if k in KEEP_KEYS}
        out.add_episode(ep)
        done += 1
        print(f'\r  [{done}/{total_eps}] User ep {i}', end='', flush=True)
    print(f'\r  [{done}/{total_eps}] User done      ')

    # Integrity check
    assert out.n_episodes == total_eps
    assert int(out.episode_ends[-1]) == out.n_steps
    ends = out.episode_ends[:]
    assert np.all(np.diff(ends) > 0), 'episode_ends not strictly increasing'
    print('  Integrity check passed.')

    # Write ZipStore
    print(f'\nWriting → {args.output} ...')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zarr.ZipStore(str(out_path), mode='w') as zs:
        zarr.copy_store(mem_store, zs)

    ep_lens = out.episode_lengths
    print(f'\nResult:')
    print(f'  Episodes : {out.n_episodes}  ({stanford_buf.n_episodes} Stanford + {user_buf.n_episodes} user)')
    print(f'  Frames   : {out.n_steps}')
    print(f'  Ep len   : min={ep_lens.min()}  max={ep_lens.max()}  mean={ep_lens.mean():.1f}')
    print(f'  Keys     : {sorted(out.keys())}')
    print(f'  Output   : {args.output}')


if __name__ == '__main__':
    main()
