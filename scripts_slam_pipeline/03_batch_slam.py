"""
python scripts_slam_pipeline/03_batch_slam.py -i data_workspace/SEHWAN_CUP_0316/demos -np -n 16
  # YAML is auto-detected from mapping dir (inherits whatever was used in step 02)
  # Override: -s /map/gopro_hero12_fisheye_setting_v1_720.yaml
"""

# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
ORIG_DIR = os.getcwd()
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import subprocess
import multiprocessing
import concurrent.futures
from tqdm import tqdm
import cv2
import av
import numpy as np
from umi.common.cv_util import draw_predefined_mask


# %%
def runner(cmd, cwd, stdout_path, stderr_path, timeout, **kwargs):
    try:
        return subprocess.run(cmd,                       
            cwd=str(cwd),
            stdout=stdout_path.open('w'),
            stderr=stderr_path.open('w'),
            timeout=timeout,
            **kwargs)
    except subprocess.TimeoutExpired as e:
        return e


# %%
@click.command()
@click.option('-i', '--input_dir', required=True, help='Directory for demos folder')
@click.option('-m', '--map_path', default=None, help='ORB_SLAM3 *.osa map atlas file')
@click.option('-d', '--docker_image', default="chicheng/orb_slam3:latest")
@click.option('-n', '--num_workers', type=int, default=None)
@click.option('-ml', '--max_lost_frames', type=int, default=60)
@click.option('-tm', '--timeout_multiple', type=float, default=16, help='timeout_multiple * duration = timeout')
@click.option('-np', '--no_docker_pull', is_flag=True, default=False, help="pull docker image from docker hub")
@click.option('-s', '--setting', default=None, help="Override SLAM settings YAML path (inside container, e.g. /data/gopro_hero12_fisheye_setting_v1.yaml)")
@click.option('-nm', '--no_mask', is_flag=True, default=False, help="Disable slam_mask (use full image for feature extraction)")
def main(input_dir, map_path, docker_image, num_workers, max_lost_frames, timeout_multiple, no_docker_pull, setting, no_mask):
    input_dir = pathlib.Path(os.path.expanduser(input_dir))
    if not input_dir.is_absolute():
        input_dir = pathlib.Path(ORIG_DIR).joinpath(input_dir)
    input_dir = input_dir.resolve()
    input_video_dirs = [x.parent for x in input_dir.glob('demo*/raw_video.mp4')]
    input_video_dirs += [x.parent for x in input_dir.glob('map*/raw_video.mp4')]
    print(f'Found {len(input_video_dirs)} video dirs')
    
    if map_path is None:
        map_path = input_dir.joinpath('mapping', 'map_atlas.osa')
    else:
        map_path = pathlib.Path(os.path.expanduser(map_path))
        if not map_path.is_absolute():
            map_path = pathlib.Path(ORIG_DIR).joinpath(map_path)
        map_path = map_path.resolve()
    assert map_path.is_file()

    # Auto-detect YAML from mapping dir when not explicitly provided.
    # Step 02 auto-copies the YAML to the mapping dir, so we inherit whatever was used for mapping.
    if setting is None:
        mapping_dir = map_path.parent
        yaml_files = sorted(mapping_dir.glob('*.yaml'))
        if yaml_files:
            # prefer 720p YAML (faster, more reliable relocalization)
            yaml_720p = [y for y in yaml_files if '720' in y.name]
            chosen_yaml = yaml_720p[0] if yaml_720p else yaml_files[0]
            setting = f'/map/{chosen_yaml.name}'
            print(f"Auto-detected SLAM settings from mapping dir: {setting}")
        else:
            setting = '/ORB_SLAM3/Examples/Monocular-Inertial/gopro10_maxlens_fisheye_setting_v1_720.yaml'
            print(f"No YAML found in mapping dir, falling back to: {setting}")

    # Determine mask resolution from the YAML (Camera.width / Camera.height)
    mask_w, mask_h = 2704, 2028  # fallback
    yaml_local_path = None
    if setting.startswith('/map/'):
        yaml_local_path = map_path.parent / setting[len('/map/'):]
    elif setting.startswith('/data/'):
        # will be resolved per video_dir below
        pass
    if yaml_local_path and yaml_local_path.is_file():
        fs = cv2.FileStorage(str(yaml_local_path), cv2.FileStorage_READ)
        if fs.isOpened():
            w = fs.getNode('Camera.width').real()
            h = fs.getNode('Camera.height').real()
            if w > 0 and h > 0:
                mask_w, mask_h = int(w), int(h)
        fs.release()

    if num_workers is None:
        num_workers = multiprocessing.cpu_count() // 2

    # pull docker
    if not no_docker_pull:
        print(f"Pulling docker image {docker_image}")
        cmd = [
            'docker',
            'pull',
            docker_image
        ]
        p = subprocess.run(cmd)
        if p.returncode != 0:
            print("Docker pull failed!")
            exit(1)

    with tqdm(total=len(input_video_dirs)) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            for video_dir in tqdm(input_video_dirs):
                video_dir = video_dir.absolute()
                if video_dir.joinpath('camera_trajectory.csv').is_file():
                    print(f"camera_trajectory.csv already exists, skipping {video_dir.name}")
                    continue
                
                # softlink won't work in bind volume
                mount_target = pathlib.Path('/data')
                csv_path = mount_target.joinpath('camera_trajectory.csv')
                video_path = mount_target.joinpath('raw_video.mp4')
                json_path = mount_target.joinpath('imu_data.json')
                mask_path = mount_target.joinpath('slam_mask.png')
                mask_write_path = video_dir.joinpath('slam_mask.png')
                
                # find video duration
                with av.open(str(video_dir.joinpath('raw_video.mp4').absolute())) as container:
                    video = container.streams.video[0]
                    duration_sec = float(video.duration * video.time_base)
                timeout = duration_sec * timeout_multiple
                
                if not no_mask:
                    slam_mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
                    slam_mask = draw_predefined_mask(
                        slam_mask, color=255, mirror=True, gripper=False, finger=True)
                    cv2.imwrite(str(mask_write_path.absolute()), slam_mask)

                map_mount_source = map_path
                map_mount_target = pathlib.Path('/map').joinpath(map_mount_source.name)

                # run SLAM
                cmd = [
                    'docker',
                    'run',
                    '--rm', # delete after finish
                    '--volume', str(video_dir) + ':' + '/data',
                    '--volume', str(map_mount_source.parent) + ':' + str(map_mount_target.parent),
                    docker_image,
                    '/ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam',
                    '--vocabulary', '/ORB_SLAM3/Vocabulary/ORBvoc.txt',
                    '--setting', setting if setting else '/ORB_SLAM3/Examples/Monocular-Inertial/gopro10_maxlens_fisheye_setting_v1_720.yaml',
                    '--input_video', str(video_path),
                    '--input_imu_json', str(json_path),
                    '--output_trajectory_csv', str(csv_path),
                    '--load_map', str(map_mount_target),
                    '--max_lost_frames', str(max_lost_frames)
                ]
                if not no_mask:
                    cmd.extend(['--mask_img', str(mask_path)])

                stdout_path = video_dir.joinpath('slam_stdout.txt')
                stderr_path = video_dir.joinpath('slam_stderr.txt')

                if len(futures) >= num_workers:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    pbar.update(len(completed))

                futures.add(executor.submit(runner,
                    cmd, str(video_dir), stdout_path, stderr_path, timeout))
                # print(' '.join(cmd))

            completed, futures = concurrent.futures.wait(futures)
            pbar.update(len(completed))

    # Print quality summary
    import csv as _csv
    all_dirs = [x.parent for x in input_dir.glob('demo*/raw_video.mp4')]
    all_dirs += [x.parent for x in input_dir.glob('map*/raw_video.mp4')]
    n_total = len(all_dirs)
    n_ok, n_fail, n_skip = 0, 0, 0
    print(f"\n{'='*50}")
    print(f"Step 03 Localization Summary")
    print(f"{'='*50}")
    for vd in sorted(all_dirs):
        traj = vd.joinpath('camera_trajectory.csv')
        if not traj.is_file():
            n_fail += 1
            continue
        with open(traj) as f:
            rows = list(_csv.DictReader(f))
        if not rows:
            n_fail += 1
            continue
        total = len(rows)
        tracked = sum(1 for r in rows if r.get('is_lost') == 'false')
        pct = 100 * tracked / total if total > 0 else 0
        if pct >= 95:
            n_ok += 1
        else:
            n_fail += 1
        print(f"  {vd.name}: {tracked}/{total} ({pct:.0f}%) {'✓' if pct >= 95 else '✗'}")
    print(f"{'='*50}")
    print(f"  Good (>=95%): {n_ok}/{n_total}  |  Poor: {n_fail}/{n_total}")
    print(f"{'='*50}\n")

# %%
if __name__ == "__main__":
    main()
