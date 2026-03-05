#!/home/keti/miniforge3/envs/umi/bin/python
"""
Revised UMI SLAM pipeline for GoPro Hero12 + UR10 + Dynamixel XH540.
Runs steps 00-07 end-to-end with Hero12 camera configurations.

Usage:
    python run_slam_pipeline_revised_gopro12.py <session_dir> [<session_dir2> ...]

Options:
    -c / --calibration_dir   Path to calibration dir (default: example/calibration)
    -n / --num_workers       Number of parallel workers (default: CPU count // 2)
    -np / --no_docker_pull   Skip docker pull
    -o / --output_zarr       Output zarr path (default: <session_dir>/replay_buffer.zarr)
"""

import sys
import os

ROOT_DIR = os.path.dirname(__file__)
sys.path.append(ROOT_DIR)
ORIG_DIR = os.getcwd()
os.chdir(ROOT_DIR)

import pathlib
import click
import subprocess
import csv
import pickle
import collections
import multiprocessing
import time

# ========================
# Helpers
# ========================

def step_header(n, name):
    bar = '#' * (46 - len(name))
    print(f"\n{'#'*50}")
    print(f"## STEP {n:02d}: {name} {bar}")
    print(f"{'#'*50}")

def ok(msg):   print(f"  [OK]   {msg}")
def warn(msg): print(f"  [WARN] {msg}")
def fail(msg): print(f"  [FAIL] {msg}")

def run(cmd, label=None):
    """Run a subprocess and return CompletedProcess. Prints timing."""
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    tag = label or pathlib.Path(cmd[1]).name
    if result.returncode == 0:
        ok(f"{tag} completed in {elapsed:.1f}s")
    else:
        fail(f"{tag} exited with code {result.returncode} after {elapsed:.1f}s")
    return result

def check_slam_quality(csv_path):
    """Parse camera_trajectory.csv and print tracking stats. Returns tracked %."""
    if not pathlib.Path(csv_path).is_file():
        warn("No trajectory CSV found.")
        return 0.0
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        warn("Trajectory CSV is empty.")
        return 0.0
    total = len(rows)
    lost  = sum(1 for r in rows if r['is_lost'] == 'true')
    tracked = total - lost
    kf = sum(1 for r in rows if r['is_keyframe'] == 'true')
    pct = 100 * tracked / total
    print(f"  Tracked : {tracked}/{total} ({pct:.1f}%)")
    print(f"  Lost    : {lost}/{total} ({100*lost/total:.1f}%)")
    print(f"  KFs     : {kf}")
    if pct >= 95:   ok("Quality: EXCELLENT (>=95%)")
    elif pct >= 80: warn("Quality: ACCEPTABLE (>=80%) — consider re-recording mapping video")
    else:           fail("Quality: POOR (<80%) — re-record mapping video")
    return pct

def check_aruco(demos_dir):
    """Summarise tag detection across all demos."""
    demos_dir = pathlib.Path(demos_dir)
    pkls = list(demos_dir.glob('*/tag_detection.pkl'))
    if not pkls:
        warn("No tag_detection.pkl files found.")
        return
    total_dirs = len(pkls)
    finger_ok = 0
    table_ok  = 0
    for pkl in pkls:
        data = pickle.load(open(pkl, 'rb'))
        n = len(data)
        if n == 0:
            continue
        counts = collections.Counter()
        for frame in data:
            for tid in frame['tag_dict']:
                counts[tid] += 1
        # finger markers 0 and 1
        f0 = counts.get(0, 0) / n
        f1 = counts.get(1, 0) / n
        if f0 >= 0.8 and f1 >= 0.8:
            finger_ok += 1
        # table marker 13
        t13 = counts.get(13, 0) / n
        if t13 >= 0.1:
            table_ok += 1
    print(f"  Dirs processed     : {total_dirs}")
    print(f"  Finger tags OK     : {finger_ok}/{total_dirs} (>=80% detection)")
    print(f"  Table tag (ID 13)  : {table_ok}/{total_dirs} dirs with any detection")
    if finger_ok == 0:
        warn("No finger tags detected — check finger marker placement (IDs 0 and 1)")
    else:
        ok(f"Finger tags detected in {finger_ok} dirs")

def check_dataset_plan(plan_path):
    """Report episode count from dataset_plan.pkl."""
    plan_path = pathlib.Path(plan_path)
    if not plan_path.is_file():
        fail("dataset_plan.pkl not found.")
        return 0
    plan = pickle.load(open(plan_path, 'rb'))
    n = len(plan)
    print(f"  Episodes planned   : {n}")
    if n > 0:
        ok(f"{n} episodes ready for replay buffer generation")
    else:
        fail("0 episodes — check SLAM tracking and ArUco detection")
    return n

# ========================
# Main
# ========================

@click.command()
@click.argument('session_dir', nargs=-1)
@click.option('-c',  '--calibration_dir',  default=None,  help='Calibration dir (default: example/calibration)')
@click.option('-n',  '--num_workers',      default=None,  type=int, help='Parallel workers')
@click.option('-p', '--docker_pull',       is_flag=True,  default=False, help='Pull docker image before running (default: skip pull)')
@click.option('-o',  '--output_zarr',      default=None,  help='Output zarr path')
def main(session_dir, calibration_dir, num_workers, docker_pull, output_zarr):
    no_docker_pull = not docker_pull

    script_dir = pathlib.Path(__file__).parent.joinpath('scripts_slam_pipeline')

    # --- calibration dir ---
    if calibration_dir is None:
        calibration_dir = pathlib.Path(__file__).parent.joinpath('example', 'calibration')
    else:
        calibration_dir = pathlib.Path(calibration_dir)
    assert calibration_dir.is_dir(), f"Calibration dir not found: {calibration_dir}"

    # --- Hero12 specific config ---
    camera_intrinsics = calibration_dir.joinpath('gopro_hero12_intrinsics_2_7k.json')
    aruco_config      = calibration_dir.joinpath('aruco_config.yaml')
    # Hero12 SLAM YAML — stored in assets/, mounted into Docker as /data/
    hero12_yaml_host  = pathlib.Path(__file__).parent.joinpath('assets', 'gopro_hero12_fisheye_setting_v1.yaml')
    assert camera_intrinsics.is_file(), f"Hero12 intrinsics not found: {camera_intrinsics}"
    assert aruco_config.is_file(),      f"ArUco config not found: {aruco_config}"
    assert hero12_yaml_host.is_file(),  f"Hero12 YAML not found: {hero12_yaml_host}"

    if num_workers is None:
        num_workers = multiprocessing.cpu_count() // 2

    python = sys.executable  # use same python that launched this script

    for session in session_dir:
        session = pathlib.Path(os.path.expanduser(session))
        if not session.is_absolute():
            session = pathlib.Path(ORIG_DIR).joinpath(session)
        session = session.resolve()

        print(f"\n{'='*50}")
        print(f"  Session : {session}")
        print(f"  YAML    : {hero12_yaml_host.name}")
        print(f"  Intr    : {camera_intrinsics.name}")
        print(f"  Workers : {num_workers}")
        print(f"{'='*50}")

        demo_dir    = session.joinpath('demos')
        mapping_dir = demo_dir.joinpath('mapping')
        map_path    = mapping_dir.joinpath('map_atlas.osa')
        plan_path   = session.joinpath('dataset_plan.pkl')
        zarr_out    = pathlib.Path(output_zarr) if output_zarr else session.joinpath('replay_buffer.zarr')

        # ── STEP 00 ──────────────────────────────────────────────
        step_header(0, '00_process_videos')
        script = script_dir.joinpath('00_process_videos.py')
        result = run([python, str(script), str(session)], '00_process_videos')
        if result.returncode != 0:
            fail("Step 00 failed — aborting session.")
            continue

        # ── STEP 01 ──────────────────────────────────────────────
        step_header(1, '01_extract_gopro_imu')
        script = script_dir.joinpath('01_extract_gopro_imu.py')
        result = run([python, str(script), str(session)], '01_extract_gopro_imu')
        if result.returncode != 0:
            fail("Step 01 failed — aborting session.")
            continue
        imu_files = list(demo_dir.glob('*/imu_data.json'))
        print(f"  IMU extracted for : {len(imu_files)} dirs")

        # ── STEP 02 ──────────────────────────────────────────────
        step_header(2, '02_create_map')
        script = script_dir.joinpath('02_create_map.py')
        assert mapping_dir.is_dir(), f"mapping/ dir not found: {mapping_dir}"
        # Copy Hero12 YAML into mapping dir so Docker can mount it via /data
        import shutil
        yaml_in_mapping = mapping_dir.joinpath(hero12_yaml_host.name)
        shutil.copy2(str(hero12_yaml_host), str(yaml_in_mapping))
        cmd = [python, str(script),
               '--input_dir', str(mapping_dir),
               '--map_path',  str(map_path),
               '--setting',   f'/data/{hero12_yaml_host.name}']
        if no_docker_pull:
            cmd.append('--no_docker_pull')
        result = run(cmd, '02_create_map')
        print("  --- Mapping SLAM quality ---")
        map_csv = mapping_dir.joinpath('mapping_camera_trajectory.csv')
        map_quality = check_slam_quality(map_csv)
        if map_quality < 80:
            fail("Mapping quality too low — re-record mapping video before continuing.")
            continue

        # ── STEP 03 ──────────────────────────────────────────────
        step_header(3, '03_batch_slam')
        script = script_dir.joinpath('03_batch_slam.py')
        cmd = [python, str(script),
               '--input_dir', str(demo_dir),
               '--map_path',  str(map_path),
               '--setting',   f'/map/{hero12_yaml_host.name}',
               '--num_workers', str(num_workers)]
        if no_docker_pull:
            cmd.append('--no_docker_pull')
        result = run(cmd, '03_batch_slam')
        # Summarise per-demo tracking
        slam_csvs = list(demo_dir.glob('demo_*/camera_trajectory.csv'))
        if slam_csvs:
            tracked_pcts = []
            for csv_p in slam_csvs:
                with open(csv_p) as f:
                    rows = list(csv.DictReader(f))
                n = len(rows)
                if n == 0: continue
                pct = 100 * sum(1 for r in rows if r['is_lost'] == 'false') / n
                tracked_pcts.append(pct)
            avg = sum(tracked_pcts) / len(tracked_pcts) if tracked_pcts else 0
            good = sum(1 for p in tracked_pcts if p >= 95)
            print(f"  Demo episodes     : {len(slam_csvs)}")
            print(f"  Avg tracking      : {avg:.1f}%")
            print(f"  Good (>=95%)      : {good}/{len(slam_csvs)}")
            if good < len(slam_csvs) * 0.5:
                warn("Less than 50% of demos tracked well — consider re-recording mapping video")
            else:
                ok(f"{good}/{len(slam_csvs)} demos with excellent tracking")

        # ── STEP 04 ──────────────────────────────────────────────
        step_header(4, '04_detect_aruco')
        script = script_dir.joinpath('04_detect_aruco.py')
        cmd = [python, str(script),
               '--input_dir',         str(demo_dir),
               '--camera_intrinsics', str(camera_intrinsics),
               '--aruco_yaml',        str(aruco_config),
               '--num_workers',       str(num_workers)]
        result = run(cmd, '04_detect_aruco')
        print("  --- ArUco detection summary ---")
        check_aruco(demo_dir)

        # ── STEP 05 ──────────────────────────────────────────────
        step_header(5, '05_run_calibrations')
        script = script_dir.joinpath('05_run_calibrations.py')
        result = run([python, str(script), str(session)], '05_run_calibrations')
        slam_tag_path = mapping_dir.joinpath('tx_slam_tag.json')
        if slam_tag_path.is_file():
            ok(f"tx_slam_tag.json generated")
        else:
            warn("tx_slam_tag.json not found — table marker (ID 13) may not have been visible")
        gripper_ranges = list(demo_dir.glob('gripper_calibration_*/gripper_range.json'))
        print(f"  Gripper ranges    : {len(gripper_ranges)} calibrated")

        # ── STEP 06 ──────────────────────────────────────────────
        step_header(6, '06_generate_dataset_plan')
        script = script_dir.joinpath('06_generate_dataset_plan.py')
        result = run([python, str(script), '--input', str(session)], '06_generate_dataset_plan')
        print("  --- Dataset plan ---")
        n_episodes = check_dataset_plan(plan_path)
        if n_episodes == 0:
            fail("No episodes generated — aborting before zarr generation.")
            continue

        # ── STEP 07 ──────────────────────────────────────────────
        step_header(7, '07_generate_replay_buffer')
        script = script_dir.joinpath('07_generate_replay_buffer.py')
        cmd = [python, str(script),
               '--output', str(zarr_out),
               '--num_workers', str(num_workers),
               str(session)]
        result = run(cmd, '07_generate_replay_buffer')
        if zarr_out.is_file() or zarr_out.is_dir():
            size_mb = sum(f.stat().st_size for f in zarr_out.rglob('*') if f.is_file()) / 1e6 \
                if zarr_out.is_dir() else zarr_out.stat().st_size / 1e6
            ok(f"replay_buffer.zarr saved ({size_mb:.0f} MB)")
        else:
            fail("replay_buffer.zarr not found after step 07")

        # ── FINAL SUMMARY ────────────────────────────────────────
        print(f"\n{'='*50}")
        print(f"  PIPELINE COMPLETE: {session.name}")
        print(f"  Episodes : {n_episodes}")
        print(f"  Zarr     : {zarr_out}")
        print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
