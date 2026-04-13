"""
python scripts_slam_pipeline/revert.py data_workspace/DYNAMIC_TOSSING/0331

Reverts the directory structure created by 00_process_videos.py.
Moves raw_video.mp4 files from demos/ subdirectories back into raw_videos/
with their original names:
  demos/mapping/raw_video.mp4                          → raw_videos/mapping.mp4
  demos/gripper_calibration_<serial>_<time>/raw_video.mp4 → raw_videos/gripper_calibration.mp4
  demos/demo_<serial>_<time>/raw_video.mp4             → raw_videos/<serial>_<time>.mp4

Also removes the demos/ directory and any symlinks in raw_videos/.
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import pathlib
import click
import shutil


@click.command(help='Revert session directories from demos/ structure back to raw_videos/')
@click.argument('session_dir', nargs=-1)
@click.option('--dry-run', is_flag=True, default=False, help='Print what would be done without moving files')
def main(session_dir, dry_run):
    for session in session_dir:
        session = pathlib.Path(os.path.expanduser(session)).absolute()
        demos_dir = session.joinpath('demos')
        raw_videos_dir = session.joinpath('raw_videos')

        if not demos_dir.is_dir():
            print(f"No demos/ directory found in {session}, skipping.")
            continue

        # Create raw_videos dir if needed
        if not dry_run:
            raw_videos_dir.mkdir(exist_ok=True)

        # Remove symlinks in raw_videos/ (created by 00_process_videos.py)
        if raw_videos_dir.is_dir():
            for f in raw_videos_dir.iterdir():
                if f.is_symlink():
                    if dry_run:
                        print(f"  Would remove symlink: {f.name}")
                    else:
                        f.unlink()
                        print(f"  Removed symlink: {f.name}")

        # Process each subdirectory in demos/
        moved = 0
        for sub_dir in sorted(demos_dir.iterdir()):
            if not sub_dir.is_dir():
                continue

            video_path = sub_dir.joinpath('raw_video.mp4')
            if not video_path.is_file():
                print(f"  No raw_video.mp4 in {sub_dir.name}, skipping.")
                continue

            dir_name = sub_dir.name

            # Determine output filename
            if dir_name == 'mapping':
                out_name = 'mapping.mp4'
            elif dir_name.startswith('gripper_calibration'):
                out_name = 'gripper_calibration.mp4'
            elif dir_name.startswith('demo_'):
                # demo_C3501325765184_2026.03.31_15.48.54.160617 → C3501325765184_2026.03.31_15.48.54.160617.mp4
                out_name = dir_name[len('demo_'):] + '.mp4'
            else:
                out_name = dir_name + '.mp4'

            out_path = raw_videos_dir.joinpath(out_name)

            # Handle duplicate names
            if out_path.exists():
                stem = out_path.stem
                suffix = out_path.suffix
                counter = 2
                while out_path.exists():
                    out_path = raw_videos_dir.joinpath(f"{stem}_{counter}{suffix}")
                    counter += 1

            if dry_run:
                print(f"  {dir_name}/raw_video.mp4 → raw_videos/{out_path.name}")
            else:
                shutil.move(str(video_path), str(out_path))
                print(f"  {dir_name}/raw_video.mp4 → raw_videos/{out_path.name}")
                moved += 1

        if not dry_run:
            # Remove empty demo subdirectories
            for sub_dir in sorted(demos_dir.iterdir(), reverse=True):
                if sub_dir.is_dir():
                    # Remove all pipeline artifacts (tag_detection.pkl, camera_trajectory.csv, etc.)
                    shutil.rmtree(sub_dir)

            # Remove demos/ directory if empty
            if demos_dir.is_dir() and not any(demos_dir.iterdir()):
                demos_dir.rmdir()
                print(f"  Removed empty demos/ directory")

            # Also remove dataset_plan.pkl and replay_buffer.zarr if they exist
            for artifact in ['dataset_plan.pkl', 'replay_buffer.zarr']:
                artifact_path = session.joinpath(artifact)
                if artifact_path.exists():
                    artifact_path.unlink()
                    print(f"  Removed {artifact}")

            print(f"\nReverted {moved} videos back to raw_videos/")
        else:
            print(f"\nDry run complete. Use without --dry-run to execute.")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main.main(['--help'])
    else:
        main()
