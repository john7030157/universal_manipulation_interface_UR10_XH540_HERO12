#!/usr/bin/env python3
"""
Repair corrupted gripper_width values in a UMI replay buffer zarr.

Common corruption: data was recorded with wrong scale factor applied in
DynamixelXH540Controller (e.g., 0.11 vs 0.09 denominator mix-up),
producing values in wrong range.

Usage examples:
  # Inspect only (dry run):
  python scripts/repair_gripper_width.py data/bad.zarr --dry-run

  # Rescale: multiply all gripper values by a factor
  # E.g., if recorded with 0.09 denominator but should be 0.11:
  #   factor = 0.09 / 0.11 ≈ 0.818
  python scripts/repair_gripper_width.py data/bad.zarr --output data/fixed.zarr --scale 0.818

  # Clamp to valid range [0, 0.11] after any rescaling:
  python scripts/repair_gripper_width.py data/bad.zarr --output data/fixed.zarr --scale 1.0 --clamp 0.0 0.11

  # Remap from [src_min, src_max] to [0, 0.11]:
  python scripts/repair_gripper_width.py data/bad.zarr --output data/fixed.zarr --remap 0.0 0.09 0.0 0.11
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import zarr


GRIPPER_KEY = 'robot0_gripper_width'
EXPECTED_MIN, EXPECTED_MAX = 0.0, 0.11


def print_stats(label: str, arr: np.ndarray):
    print(f'  {label:12s}  min={arr.min():.5f}  max={arr.max():.5f}  '
          f'mean={arr.mean():.5f}  std={arr.std():.5f}')


def load_zarr(path: str):
    p = Path(path)
    if p.suffix == '.zip':
        return zarr.open(zarr.ZipStore(path, mode='r'), mode='r')
    return zarr.open(path, mode='r')


def copy_zarr(src_path: str, dst_path: str):
    """Copy zarr to output path (uncompressed) for in-place editing."""
    dst = Path(dst_path)
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    zarr.copy_store(
        zarr.ZipStore(src_path, mode='r') if src_path.endswith('.zip')
        else zarr.DirectoryStore(src_path),
        zarr.DirectoryStore(dst_path),
    )
    print(f'Copied {src_path} → {dst_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input zarr path')
    parser.add_argument('--output', default=None, help='Output zarr path (required unless --dry-run)')
    parser.add_argument('--dry-run', action='store_true', help='Print stats only, do not write')
    parser.add_argument('--scale', type=float, default=None,
                        help='Multiply gripper values by this factor')
    parser.add_argument('--remap', nargs=4, type=float, metavar=('SRC_MIN', 'SRC_MAX', 'DST_MIN', 'DST_MAX'),
                        help='Linearly remap from [src_min,src_max] to [dst_min,dst_max]')
    parser.add_argument('--clamp', nargs=2, type=float, metavar=('LOW', 'HIGH'),
                        default=None, help='Clamp values to [low, high] after other transforms')
    parser.add_argument('--gripper-key', default=GRIPPER_KEY)
    args = parser.parse_args()

    if not args.dry_run and args.output is None:
        print('ERROR: --output required unless --dry-run')
        return 1

    root = load_zarr(args.input)
    gw = root['data'][args.gripper_key][:]

    print(f'\nInput: {args.input}')
    print_stats('before', gw)

    if args.dry_run:
        if gw.max() > EXPECTED_MAX * 1.5 or gw.min() < -0.01:
            print(f'\n*** VALUES OUT OF EXPECTED RANGE [{EXPECTED_MIN}, {EXPECTED_MAX}] ***')
            print('Suggested fix: determine the scale factor and re-run with --scale or --remap')
        return 0

    gw_fixed = gw.copy().astype(np.float32)

    if args.remap:
        src_min, src_max, dst_min, dst_max = args.remap
        gw_fixed = (gw_fixed - src_min) / (src_max - src_min) * (dst_max - dst_min) + dst_min
        print(f'  remapped [{src_min}, {src_max}] → [{dst_min}, {dst_max}]')

    if args.scale is not None:
        gw_fixed = gw_fixed * args.scale
        print(f'  scaled by {args.scale}')

    if args.clamp is not None:
        gw_fixed = np.clip(gw_fixed, args.clamp[0], args.clamp[1])
        print(f'  clamped to [{args.clamp[0]}, {args.clamp[1]}]')

    print_stats('after ', gw_fixed)

    # Write output
    copy_zarr(args.input, args.output)
    out_root = zarr.open(args.output, mode='r+')
    out_root['data'][args.gripper_key][:] = gw_fixed
    print(f'\nWrote fixed zarr to {args.output}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
