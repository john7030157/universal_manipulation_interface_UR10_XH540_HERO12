#!/usr/bin/env python3
"""
Inspect a UMI replay buffer zarr — episode count, key shapes, gripper stats.

Usage:
  python scripts/inspect_zarr.py PATH_TO_ZARR [--gripper-key robot0_gripper_width]
"""

import argparse
import sys

import numpy as np
import zarr


def inspect(zarr_path: str, gripper_key: str = 'robot0_gripper_width'):
    print(f'\n=== {zarr_path} ===')
    try:
        store = zarr.ZipStore(zarr_path, mode='r')
    except Exception:
        store = zarr_path  # uncompressed

    root = zarr.open(store, mode='r')

    # Episode count
    episode_ends = root['meta/episode_ends'][:]
    n_episodes = len(episode_ends)
    n_steps = int(episode_ends[-1]) if n_episodes > 0 else 0
    print(f'\nEpisodes : {n_episodes}')
    print(f'Total steps : {n_steps}')

    # Episode length distribution
    starts = np.concatenate([[0], episode_ends[:-1]])
    lengths = episode_ends - starts
    print(f'Episode length  min={lengths.min()}  max={lengths.max()}  '
          f'mean={lengths.mean():.1f}  median={np.median(lengths):.1f}')

    # All keys in data group
    print('\nData keys:')
    for key in sorted(root['data'].keys()):
        arr = root['data'][key]
        print(f'  {key:45s}  shape={arr.shape}  dtype={arr.dtype}')

    # Gripper width stats
    if gripper_key in root['data']:
        gw = root['data'][gripper_key][:]
        print(f'\nGripper ({gripper_key}):')
        print(f'  min={gw.min():.4f}  max={gw.max():.4f}  '
              f'mean={gw.mean():.4f}  std={gw.std():.4f}')
        # Sanity check: XH540 physical range is 0-0.09m; policy trains in 0-0.11m
        if gw.max() > 0.2 or gw.min() < -0.01:
            print(f'  *** WARNING: gripper values out of expected 0-0.11 range! '
                  f'Likely corrupted (wrong scale or units). ***')
        elif gw.max() < 0.05:
            print(f'  *** WARNING: gripper max is very small ({gw.max():.4f}). '
                  f'May be stuck closed or wrong scale. ***')
        else:
            print(f'  Gripper range looks reasonable (expected ~0 to 0.09-0.11 m).')

        # Per-episode gripper range
        print(f'\n  Per-episode gripper range (first 10 episodes):')
        for i in range(min(10, n_episodes)):
            s = int(starts[i])
            e = int(episode_ends[i])
            ep_gw = gw[s:e]
            flag = ' ***SUSPECT***' if (ep_gw.max() > 0.2 or ep_gw.max() < 0.02) else ''
            print(f'    ep {i:3d}: min={ep_gw.min():.4f}  max={ep_gw.max():.4f}{flag}')
    else:
        print(f'\n  Key "{gripper_key}" not found in data.')

    # EEF position range (sanity check workspace bounds)
    pos_key = 'robot0_eef_pos'
    if pos_key in root['data']:
        pos = root['data'][pos_key][:]
        print(f'\nEEF position range (XYZ):')
        for i, axis in enumerate('XYZ'):
            print(f'  {axis}: [{pos[:, i].min():.3f}, {pos[:, i].max():.3f}] m')

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('zarr_path', nargs='+', help='Path(s) to zarr file')
    parser.add_argument('--gripper-key', default='robot0_gripper_width')
    args = parser.parse_args()

    for path in args.zarr_path:
        try:
            inspect(path, gripper_key=args.gripper_key)
        except Exception as e:
            print(f'ERROR reading {path}: {e}', file=sys.stderr)


if __name__ == '__main__':
    main()
