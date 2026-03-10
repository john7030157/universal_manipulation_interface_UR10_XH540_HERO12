"""
python scripts_slam_pipeline/02_create_map.py -i data_workspace/Cup_Placement_Test_Session/demos/mapping -np -s /data/gopro_hero12_fisheye_setting_v1.yaml -e 5
"""

# %%
import sys
import os
 
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import subprocess
import json as _json
import numpy as np
import cv2
from umi.common.cv_util import draw_predefined_mask

# %%
import shutil as _shutil
import csv as _csv

def _read_quality(csv_path, stdout_path):
    """Return (tracked_pct, tracked, total, kf, n_maps) or None if CSV missing/empty."""
    if not csv_path.is_file():
        return None
    with open(csv_path) as f:
        rows = list(_csv.DictReader(f))
    total = len(rows)
    if total == 0:
        return None
    lost = sum(1 for r in rows if r['is_lost'] == 'true')
    tracked = total - lost
    kf = sum(1 for r in rows if r['is_keyframe'] == 'true')
    n_maps = 0
    for line in stdout_path.read_text().splitlines():
        if 'maps in the atlas' in line:
            try:
                n_maps = int(line.strip().split()[2])
            except Exception:
                pass
    return (100 * tracked / total, tracked, total, kf, n_maps)

def _print_quality(label, q):
    tracked_pct, tracked, total, kf, n_maps = q
    print(f"\n=== SLAM Mapping Quality ({label}) ===")
    print(f"{'Total frames':<20}: {total}")
    print(f"{'Tracked':<20}: {tracked} ({tracked_pct:.1f}%)")
    print(f"{'Lost':<20}: {total - tracked} ({100 - tracked_pct:.1f}%)")
    print(f"{'Keyframes':<20}: {kf}")
    print(f"{'Maps in atlas':<20}: {n_maps}")
    if tracked_pct >= 95:
        print("Quality: EXCELLENT (>=95% tracked)")
    elif tracked_pct >= 80:
        print("Quality: ACCEPTABLE (>=80% tracked)")
    else:
        print("Quality: POOR (<80% tracked) — consider re-recording mapping video")
    print("============================\n")


@click.command()
@click.option('-i', '--input_dir', required=True, help='Directory for mapping video')
@click.option('-m', '--map_path', default=None, help='ORB_SLAM3 *.osa map atlas file')
@click.option('-d', '--docker_image', default="chicheng/orb_slam3:latest")
@click.option('-np', '--no_docker_pull', is_flag=True, default=False, help="pull docker image from docker hub")
@click.option('-nm', '--no_mask', is_flag=True, default=False, help="Whether to mask out gripper and mirrors. Set if map is created with bare GoPro no on gripper.")
@click.option('-l', '--local', is_flag=True, default=False, help="Use local ORB_SLAM3 binary instead of Docker")
@click.option('-od', '--orb_slam_dir', default=None, help="Path to local ORB_SLAM3 directory (used with --local)")
@click.option('-s', '--setting', default=None, help="Override SLAM settings YAML path")
@click.option('-e', '--epochs', default=1, type=int, help="Number of SLAM attempts; best result is kept (default: 1)")
def main(input_dir, map_path, docker_image, no_docker_pull, no_mask, local, orb_slam_dir, setting, epochs):
    video_dir = pathlib.Path(os.path.expanduser(input_dir)).absolute()
    for fn in ['raw_video.mp4', 'imu_data.json']:
        assert video_dir.joinpath(fn).is_file()

    if map_path is None:
        map_path = video_dir.joinpath('map_atlas.osa')
    else:
        map_path = pathlib.Path(os.path.expanduser(map_path)).absolute()
    map_path.parent.mkdir(parents=True, exist_ok=True)

    # detect video resolution to pick right mask and YAML
    probe = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams',
         str(video_dir.joinpath('raw_video.mp4'))],
        capture_output=True, text=True)
    streams = _json.loads(probe.stdout)['streams']
    vid = next(s for s in streams if s.get('codec_type') == 'video')
    vid_w, vid_h = int(vid['width']), int(vid['height'])
    print(f"Video resolution: {vid_w}x{vid_h}")

    if not no_mask:
        mask_write_path = video_dir.joinpath('slam_mask.png')
        slam_mask = np.zeros((vid_h, vid_w), dtype=np.uint8)
        slam_mask = draw_predefined_mask(
            slam_mask, color=255, mirror=True, gripper=False, finger=True)
        cv2.imwrite(str(mask_write_path.absolute()), slam_mask)

    map_mount_source = pathlib.Path(map_path)

    if local:
        # Use local ORB_SLAM3 binary
        if orb_slam_dir is None:
            # default: look for ORB_SLAM3 next to the UMI repo
            orb_slam_dir = pathlib.Path(ROOT_DIR).parent.joinpath('ORB_SLAM3')
        orb_slam_dir = pathlib.Path(orb_slam_dir).absolute()
        gopro_slam_bin = orb_slam_dir.joinpath('Examples', 'Monocular-Inertial', 'gopro_slam')
        vocabulary = orb_slam_dir.joinpath('Vocabulary', 'ORBvoc.txt')
        assert gopro_slam_bin.is_file(), f"gopro_slam binary not found: {gopro_slam_bin}\nBuild ORB_SLAM3 first."
        assert vocabulary.is_file(), f"Vocabulary not found: {vocabulary}\nExtract ORBvoc.txt.tar.gz first."

        if setting is None:
            setting = str(orb_slam_dir.joinpath(
                'Examples', 'Monocular-Inertial', 'gopro_hero12_fisheye_setting_v1.yaml'))

        cmd = [
            str(gopro_slam_bin),
            '--vocabulary', str(vocabulary),
            '--setting', setting,
            '--input_video', str(video_dir.joinpath('raw_video.mp4')),
            '--input_imu_json', str(video_dir.joinpath('imu_data.json')),
            '--output_trajectory_csv', str(video_dir.joinpath('mapping_camera_trajectory.csv')),
            '--save_map', str(map_mount_source)
        ]
        if not no_mask:
            cmd.extend(['--mask_img', str(video_dir.joinpath('slam_mask.png'))])
    else:
        # Docker mode
        map_mount_target = pathlib.Path('/map').joinpath(map_mount_source.name)
        mount_target = pathlib.Path('/data')
        csv_path = mount_target.joinpath('mapping_camera_trajectory.csv')
        video_path = mount_target.joinpath('raw_video.mp4')
        json_path = mount_target.joinpath('imu_data.json')
        mask_path = mount_target.joinpath('slam_mask.png')

        # pick the right YAML based on resolution
        if setting is None:
            if vid_w == 2704 and vid_h == 2028:
                slam_yaml = 'gopro_hero12_fisheye_setting_v1.yaml'
            else:
                slam_yaml = 'gopro_hero12_fisheye_setting_v1.yaml'
            setting = f'/ORB_SLAM3/Examples/Monocular-Inertial/{slam_yaml}'
        print(f"Using SLAM settings: {setting}")

        # if setting is a /data/ path, the YAML must exist in video_dir (mounted as /data).
        # auto-copy it from known locations if missing.
        if setting.startswith('/data/'):
            yaml_name = setting[len('/data/'):]
            yaml_dst = video_dir.joinpath(yaml_name)
            if not yaml_dst.is_file():
                search_dirs = [
                    pathlib.Path(ROOT_DIR).joinpath('assets'),
                    pathlib.Path(ROOT_DIR).parent.joinpath('ORB_SLAM3', 'Examples', 'Monocular-Inertial'),
                ]
                for src_dir in search_dirs:
                    candidate = src_dir.joinpath(yaml_name)
                    if candidate.is_file():
                        _shutil.copy2(str(candidate), str(yaml_dst))
                        print(f"Auto-copied YAML: {candidate} -> {yaml_dst}")
                        break
                else:
                    raise FileNotFoundError(
                        f"YAML '{yaml_name}' not found in video_dir or any of: {search_dirs}\n"
                        f"Copy it manually: cp <your_yaml> {yaml_dst}")

        # pull docker
        if not no_docker_pull:
            print(f"Pulling docker image {docker_image}")
            p = subprocess.run(['docker', 'pull', docker_image])
            if p.returncode != 0:
                print("Docker pull failed!")
                exit(1)

        cmd = [
            'docker', 'run', '--rm',
            '--volume', str(video_dir) + ':' + '/data',
            '--volume', str(map_mount_source.parent) + ':' + str(map_mount_target.parent),
            docker_image,
            '/ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam',
            '--vocabulary', '/ORB_SLAM3/Vocabulary/ORBvoc.txt',
            '--setting', setting,
            '--input_video', str(video_path),
            '--input_imu_json', str(json_path),
            '--output_trajectory_csv', str(csv_path),
            '--save_map', str(map_mount_target)
        ]
        if not no_mask:
            cmd.extend(['--mask_img', str(mask_path)])

    stdout_path = video_dir.joinpath('slam_stdout.txt')
    stderr_path = video_dir.joinpath('slam_stderr.txt')
    csv_result_path = video_dir.joinpath('mapping_camera_trajectory.csv')

    # temp paths for the best result so far
    best_map_path  = map_path.parent.joinpath('map_atlas_best.osa')
    best_csv_path  = video_dir.joinpath('mapping_camera_trajectory_best.csv')
    best_stdout    = video_dir.joinpath('slam_stdout_best.txt')
    best_stderr    = video_dir.joinpath('slam_stderr_best.txt')

    best_pct = -1.0
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        if epochs > 1:
            print(f"\n{'─'*40}")
            print(f"  Epoch {epoch}/{epochs}")
            print(f"{'─'*40}")

        subprocess.run(
            cmd,
            cwd=str(video_dir),
            stdout=stdout_path.open('w'),
            stderr=stderr_path.open('w')
        )

        q = _read_quality(csv_result_path, stdout_path)
        if q is None:
            print(f"  Epoch {epoch}: no trajectory CSV — SLAM failed, skipping.")
            continue

        tracked_pct = q[0]
        _print_quality(f"epoch {epoch}/{epochs}", q)

        if tracked_pct > best_pct:
            best_pct = tracked_pct
            best_epoch = epoch
            # save this epoch's outputs as the current best
            if map_path.is_file():
                _shutil.copy2(str(map_path), str(best_map_path))
            _shutil.copy2(str(csv_result_path), str(best_csv_path))
            _shutil.copy2(str(stdout_path), str(best_stdout))
            _shutil.copy2(str(stderr_path), str(best_stderr))
            print(f"  ^ New best: {tracked_pct:.1f}% (epoch {epoch})")

    # restore best result as the final output
    if epochs > 1:
        print(f"\n{'='*40}")
        if best_epoch >= 0:
            print(f"  Best epoch : {best_epoch}/{epochs}  ({best_pct:.1f}% tracked)")
            _shutil.move(str(best_map_path),  str(map_path))
            _shutil.move(str(best_csv_path),  str(csv_result_path))
            _shutil.move(str(best_stdout),    str(stdout_path))
            _shutil.move(str(best_stderr),    str(stderr_path))
        else:
            print("  All epochs failed — no valid map produced.")
        print(f"{'='*40}\n")
    else:
        # single epoch — clean up temp files if they were accidentally created
        for p in [best_map_path, best_csv_path, best_stdout, best_stderr]:
            if p.is_file():
                p.unlink()


# %%
if __name__ == "__main__":
    main()
