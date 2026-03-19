"""
# GoPro Hero12 (our setup) — always run with miniforge3/envs/umi python:
/home/keti/miniforge3/envs/umi/bin/python scripts_slam_pipeline/04_detect_aruco.py -i data_workspace/<session>/demos -ci example/calibration/gopro_hero12_intrinsics_2_7k.json -ac example/calibration/aruco_config.yaml

# Add -redo to delete existing tag_detection.pkl and reprocess all dirs:
/home/keti/miniforge3/envs/umi/bin/python scripts_slam_pipeline/04_detect_aruco.py -i data_workspace/<session>/demos -ci example/calibration/gopro_hero12_intrinsics_2_7k.json -ac example/calibration/aruco_config.yaml -redo

# GoPro Hero10 MaxLens (Stanford example dataset):
/home/keti/miniforge3/envs/umi/bin/python scripts_slam_pipeline/04_detect_aruco.py -i external_data_workspace/cup_in_lab_mp4s/20231204/demos -ci example/calibration/gopro_intrinsics_2_7k.json -ac example/calibration/aruco_config.yaml

# Notes:
# - Uses sys.executable to spawn subprocesses (avoids missing 'av' module with system python)
# - cv_util.py uses solvePnP instead of estimatePoseSingleMarkers (broken in OpenCV 4.13)
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
import multiprocessing
import subprocess
import concurrent.futures
from tqdm import tqdm

# %%
@click.command()
@click.option('-i', '--input_dir', required=True, help='Directory for demos folder')
@click.option('-ci', '--camera_intrinsics', required=True, help='Camera intrinsics json file (2.7k)')
@click.option('-ac', '--aruco_yaml', required=True, help='Aruco config yaml file')
@click.option('-n', '--num_workers', type=int, default=None)
@click.option('-redo', '--redo', is_flag=True, default=False, help='Delete existing tag_detection.pkl and reprocess all dirs')
def main(input_dir, camera_intrinsics, aruco_yaml, num_workers, redo):
    input_dir = pathlib.Path(os.path.expanduser(input_dir))
    input_video_dirs = [x.parent for x in input_dir.glob('*/raw_video.mp4')]
    print(f'Found {len(input_video_dirs)} video dirs')

    assert os.path.isfile(camera_intrinsics)
    assert os.path.isfile(aruco_yaml)

    if redo:
        for video_dir in input_video_dirs:
            pkl_path = video_dir.absolute().joinpath('tag_detection.pkl')
            if pkl_path.is_file():
                pkl_path.unlink()
                print(f"Deleted {pkl_path}")

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    script_path = pathlib.Path(__file__).parent.parent.joinpath('scripts', 'detect_aruco.py')

    with tqdm(total=len(input_video_dirs)) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            for video_dir in tqdm(input_video_dirs):
                video_dir = video_dir.absolute()
                video_path = video_dir.joinpath('raw_video.mp4')
                pkl_path = video_dir.joinpath('tag_detection.pkl')
                if pkl_path.is_file():
                    print(f"tag_detection.pkl already exists, skipping {video_dir.name}")
                    continue

                # run SLAM
                cmd = [
                    sys.executable, script_path,
                    '--input', str(video_path),
                    '--output', str(pkl_path),
                    '--intrinsics_json', camera_intrinsics,
                    '--aruco_yaml', aruco_yaml,
                    '--num_workers', '1'
                ]

                if len(futures) >= num_workers:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    pbar.update(len(completed))

                futures.add(executor.submit(
                    lambda x: subprocess.run(x, 
                        capture_output=True), 
                    cmd))
                # futures.add(executor.submit(lambda x: print(' '.join(x)), cmd))

            completed, futures = concurrent.futures.wait(futures)            
            pbar.update(len(completed))

    n_total = len(input_video_dirs)
    n_ok = sum(1 for d in input_video_dirs if d.absolute().joinpath('tag_detection.pkl').is_file())
    print(f"\n{'='*40}")
    print(f"Step 04 ArUco Detection Summary")
    print(f"{'='*40}")
    print(f"  PKL produced : {n_ok}/{n_total}")
    print(f"  Missing      : {n_total - n_ok}/{n_total}")
    print(f"{'='*40}\n")

# %%
if __name__ == "__main__":
    main()
