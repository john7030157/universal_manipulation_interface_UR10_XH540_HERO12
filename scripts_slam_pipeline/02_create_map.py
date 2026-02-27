"""
python scripts_slam_pipeline/00_process_videos.py -i data_workspace/toss_objects/20231113/mapping
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
@click.command()
@click.option('-i', '--input_dir', required=True, help='Directory for mapping video')
@click.option('-m', '--map_path', default=None, help='ORB_SLAM3 *.osa map atlas file')
@click.option('-d', '--docker_image', default="chicheng/orb_slam3:latest")
@click.option('-np', '--no_docker_pull', is_flag=True, default=False, help="pull docker image from docker hub")
@click.option('-nm', '--no_mask', is_flag=True, default=False, help="Whether to mask out gripper and mirrors. Set if map is created with bare GoPro no on gripper.")
@click.option('-l', '--local', is_flag=True, default=False, help="Use local ORB_SLAM3 binary instead of Docker")
@click.option('-od', '--orb_slam_dir', default=None, help="Path to local ORB_SLAM3 directory (used with --local)")
@click.option('-s', '--setting', default=None, help="Override SLAM settings YAML path")
def main(input_dir, map_path, docker_image, no_docker_pull, no_mask, local, orb_slam_dir, setting):
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
                'Examples', 'Monocular-Inertial', 'gopro10_maxlens_fisheye_setting_v1.yaml'))

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
                slam_yaml = 'gopro10_maxlens_fisheye_setting_v1.yaml'
            else:
                slam_yaml = 'gopro10_maxlens_fisheye_setting_v1_720.yaml'
            setting = f'/ORB_SLAM3/Examples/Monocular-Inertial/{slam_yaml}'
        print(f"Using SLAM settings: {setting}")

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

    result = subprocess.run(
        cmd,
        cwd=str(video_dir),
        stdout=stdout_path.open('w'),
        stderr=stderr_path.open('w')
    )
    print(result)


# %%
if __name__ == "__main__":
    main()
