#!/home/keti/miniforge3/envs/umi/bin/python
"""
Final UMI SLAM pipeline for GoPro Hero12 + UR10 + Dynamixel XH540.
Runs steps 00-07 end-to-end with two-pass step 03 retry logic.

Usage:
    conda run -n umi python run_slam_pipeline_final.py <session_dir> [<session_dir2> ...]

Options:
    -c / --calibration_dir   Calibration dir (default: example/calibration)
    -n / --num_workers       Parallel workers for steps 03/04/07 (default: CPU//2)
    -p / --docker_pull       Pull docker image before running (default: skip)
    -e / --epochs            SLAM mapping attempts in step 02, best kept (default: 5)
    -ml / --max_lost_frames  Max consecutive lost frames before giving up in pass 1 (default: 60)
    -ml2 / --max_lost_frames2  Max lost frames for pass 2 retry (default: 150)
    -redo / --redo_aruco     Delete existing tag_detection.pkl and rerun step 04
    -o / --output_zarr       Output zarr path (default: <session>/replay_buffer.zarr)

Step 03 two-pass strategy:
    Pass 1 uses -ml (fast). Demos that fail or score <95% are retried in pass 2
    with -ml2. Demos that passed on pass 1 are skipped (camera_trajectory.csv exists).
    Demos with true map coverage gaps fail both passes quickly.

Key design decisions:
    - Step 02: auto-selects 720p YAML from assets/ (no -s needed)
    - Step 03: auto-inherits YAML from mapping dir (no -s needed)
    - 720p YAML (nLevels=8) required for reliable Hero12 relocalization
    - sys.executable used throughout so conda env's cv2/av are available
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
import multiprocessing
import time


# ========================
# Helpers
# ========================

def step_header(n, name):
    bar = '#' * max(1, 46 - len(name))
    print(f"\n{'#'*50}")
    print(f"## STEP {n:02d}: {name} {bar}")
    print(f"{'#'*50}")

def ok(msg):   print(f"  [OK]   {msg}")
def warn(msg): print(f"  [WARN] {msg}")
def fail(msg): print(f"  [FAIL] {msg}")

def run(cmd, label=None):
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    tag = label or pathlib.Path(cmd[1]).name
    if result.returncode == 0:
        ok(f"{tag} completed in {elapsed:.1f}s")
    else:
        fail(f"{tag} exited with code {result.returncode} after {elapsed:.1f}s")
    return result

def slam_tracking_pct(csv_path):
    """Return tracked% for a camera_trajectory.csv, or None if missing/empty."""
    p = pathlib.Path(csv_path)
    if not p.is_file():
        return None
    with open(p) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    tracked = sum(1 for r in rows if r.get('is_lost') == 'false')
    return 100.0 * tracked / len(rows)

def check_slam_quality(csv_path):
    """Print mapping SLAM quality and return tracked%. Used for step 02."""
    pct = slam_tracking_pct(csv_path)
    if pct is None:
        warn("No mapping trajectory CSV found.")
        return 0.0
    p = pathlib.Path(csv_path)
    with open(p) as f:
        rows = list(csv.DictReader(f))
    total   = len(rows)
    tracked = sum(1 for r in rows if r.get('is_lost') == 'false')
    kf      = sum(1 for r in rows if r.get('is_keyframe') == 'true')
    print(f"  Tracked : {tracked}/{total} ({pct:.1f}%)")
    print(f"  Lost    : {total-tracked}/{total} ({100-pct:.1f}%)")
    print(f"  KFs     : {kf}")
    if pct >= 95:   ok("Mapping quality: EXCELLENT (>=95%)")
    elif pct >= 80: warn("Mapping quality: ACCEPTABLE (>=80%) — consider re-recording mapping video")
    else:           fail("Mapping quality: POOR (<80%) — re-record mapping video")
    return pct

def slam_summary(demo_dir, label=""):
    """Print per-demo tracking summary. Returns (good, total) counts."""
    all_dirs = [x.parent for x in pathlib.Path(demo_dir).glob('demo*/raw_video.mp4')]
    n_total  = len(all_dirs)
    n_ok = n_fail = n_no_csv = 0
    tag = f" ({label})" if label else ""
    print(f"\n  --- Step 03 summary{tag} ---")
    for vd in sorted(all_dirs):
        pct = slam_tracking_pct(vd / 'camera_trajectory.csv')
        if pct is None:
            n_no_csv += 1
            print(f"    {vd.name}: no CSV")
        elif pct >= 95:
            n_ok += 1
            print(f"    {vd.name}: {pct:.0f}% ✓")
        else:
            n_fail += 1
            print(f"    {vd.name}: {pct:.0f}% ✗")
    print(f"  Good (>=95%): {n_ok}/{n_total}  |  Poor: {n_fail}/{n_total}  |  No CSV: {n_no_csv}/{n_total}")
    return n_ok, n_total

def find_failed_demos(demo_dir):
    """Return list of demo dirs that have no CSV or tracked% < 95%."""
    failed = []
    for vd in pathlib.Path(demo_dir).glob('demo*/'):
        if not (vd / 'raw_video.mp4').is_file():
            continue
        pct = slam_tracking_pct(vd / 'camera_trajectory.csv')
        if pct is None or pct < 95:
            failed.append(vd)
    return failed

def check_dataset_plan(plan_path):
    plan_path = pathlib.Path(plan_path)
    if not plan_path.is_file():
        fail("dataset_plan.pkl not found.")
        return 0
    plan = pickle.load(open(plan_path, 'rb'))
    n = len(plan)
    print(f"  Episodes planned : {n}")
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
@click.option('-c',   '--calibration_dir',    default=None,  help='Calibration dir (default: example/calibration)')
@click.option('-n',   '--num_workers',         default=None,  type=int, help='Parallel workers for steps 03/04/07')
@click.option('-p',   '--docker_pull',         is_flag=True,  default=False, help='Pull docker image before running')
@click.option('-l',   '--local',               is_flag=True,  default=True,  help='Use local ORB_SLAM3 binary (default: True)')
@click.option('-od',  '--orb_slam_dir',        default=None,  help='Path to dir containing gopro_slam binary (default: auto-detect)')
@click.option('-e',   '--epochs',              default=5,     type=int, help='Step 02 SLAM attempts; best kept (default: 5)')
@click.option('-ml',  '--max_lost_frames',     default=60,    type=int, help='Max consecutive lost frames, pass 1 (default: 60)')
@click.option('-ml2', '--max_lost_frames2',    default=150,   type=int, help='Max consecutive lost frames, pass 2 retry (default: 150)')
@click.option('-redo','--redo_aruco',          is_flag=True,  default=False, help='Delete existing tag_detection.pkl and rerun step 04')
@click.option('-o',   '--output_zarr',         default=None,  help='Output zarr path (default: <session>/replay_buffer.zarr)')
def main(session_dir, calibration_dir, num_workers, docker_pull, local, orb_slam_dir,
         epochs, max_lost_frames, max_lost_frames2, redo_aruco, output_zarr):

    no_docker_pull = not docker_pull
    script_dir = pathlib.Path(__file__).parent.joinpath('scripts_slam_pipeline')

    if local:
        # Binary auto-detected by 02/03 scripts — just print where we expect it
        bundled = pathlib.Path(__file__).parent.joinpath('Monocular-Inertial', 'gopro_slam')
        print(f"  Local binary : {bundled if bundled.is_file() else '(auto-detect)'}")

    if calibration_dir is None:
        calibration_dir = pathlib.Path(__file__).parent.joinpath('example', 'calibration')
    else:
        calibration_dir = pathlib.Path(calibration_dir)
    assert calibration_dir.is_dir(), f"Calibration dir not found: {calibration_dir}"

    camera_intrinsics = calibration_dir.joinpath('gopro_hero12_intrinsics_2_7k.json')
    aruco_config      = calibration_dir.joinpath('aruco_config.yaml')
    assert camera_intrinsics.is_file(), f"Hero12 intrinsics not found: {camera_intrinsics}"
    assert aruco_config.is_file(),      f"ArUco config not found: {aruco_config}"

    # Confirm 720p YAML exists in assets/ (auto-selected by step 02 and step 03)
    yaml_720p = pathlib.Path(__file__).parent.joinpath('assets', 'gopro_hero12_fisheye_setting_v1_720.yaml')
    assert yaml_720p.is_file(), f"720p YAML not found: {yaml_720p}\nCopy it to assets/ first."

    if num_workers is None:
        num_workers = multiprocessing.cpu_count() // 2

    python = sys.executable

    for session in session_dir:
        session = pathlib.Path(os.path.expanduser(session))
        if not session.is_absolute():
            session = pathlib.Path(ORIG_DIR).joinpath(session)
        session = session.resolve()

        demo_dir    = session.joinpath('demos')
        mapping_dir = demo_dir.joinpath('mapping')
        map_path    = mapping_dir.joinpath('map_atlas.osa')
        plan_path   = session.joinpath('dataset_plan.pkl')
        zarr_out    = pathlib.Path(output_zarr) if output_zarr else session.joinpath('replay_buffer.zarr')

        print(f"\n{'='*50}")
        print(f"  Session  : {session.name}")
        print(f"  YAML     : {yaml_720p.name} (auto-selected)")
        print(f"  Intr     : {camera_intrinsics.name}")
        print(f"  Workers  : {num_workers}")
        print(f"  Epochs   : {epochs}")
        print(f"  -ml pass1: {max_lost_frames}  pass2: {max_lost_frames2}")
        print(f"{'='*50}")

        # ── STEP 00 ──────────────────────────────────────────────
        step_header(0, '00_process_videos')
        result = run([python, str(script_dir / '00_process_videos.py'), str(session)], '00_process_videos')
        if result.returncode != 0:
            fail("Step 00 failed — aborting session.")
            continue

        # ── STEP 01 ──────────────────────────────────────────────
        step_header(1, '01_extract_gopro_imu')
        result = run([python, str(script_dir / '01_extract_gopro_imu.py'), str(session)], '01_extract_gopro_imu')
        if result.returncode != 0:
            fail("Step 01 failed — aborting session.")
            continue
        imu_files = list(demo_dir.glob('*/imu_data.json'))
        print(f"  IMU extracted for : {len(imu_files)} dirs")

        # ── STEP 02 ──────────────────────────────────────────────
        # No -s needed: 02_create_map.py auto-selects 720p YAML from assets/
        step_header(2, '02_create_map')
        assert mapping_dir.is_dir(), f"mapping/ dir not found: {mapping_dir}"
        cmd = [python, str(script_dir / '02_create_map.py'),
               '--input_dir', str(mapping_dir),
               '--map_path',  str(map_path),
               '--epochs',    str(epochs)]
        if local:
            cmd.append('--local')
        elif no_docker_pull:
            cmd.append('--no_docker_pull')
        result = run(cmd, '02_create_map')
        print("  --- Mapping SLAM quality ---")
        map_quality = check_slam_quality(mapping_dir / 'mapping_camera_trajectory.csv')
        if map_quality < 80:
            fail("Mapping quality too low (<80%) — re-record mapping video before continuing.")
            continue

        # ── STEP 03 — Pass 1 ─────────────────────────────────────
        # No -s needed: 03_batch_slam.py auto-detects YAML from mapping dir
        step_header(3, '03_batch_slam — Pass 1')
        cmd = [python, str(script_dir / '03_batch_slam.py'),
               '--input_dir',      str(demo_dir),
               '--map_path',       str(map_path),
               '--max_lost_frames', str(max_lost_frames),
               '--num_workers',    str(num_workers)]
        if local:
            cmd.append('--local')
        elif no_docker_pull:
            cmd.append('--no_docker_pull')
        run(cmd, '03_batch_slam pass1')
        n_ok_p1, n_total = slam_summary(demo_dir, f"pass 1, -ml {max_lost_frames}")

        # ── STEP 03 — Pass 2 (retry failed demos with higher -ml) ─
        failed_after_p1 = find_failed_demos(demo_dir)
        if failed_after_p1 and max_lost_frames2 > max_lost_frames:
            step_header(3, f'03_batch_slam — Pass 2 (retry {len(failed_after_p1)} demos, -ml {max_lost_frames2})')
            print(f"  Retrying {len(failed_after_p1)} demos with -ml {max_lost_frames2}:")
            for vd in sorted(failed_after_p1):
                csv_p = vd / 'camera_trajectory.csv'
                if csv_p.is_file():
                    csv_p.unlink()
                    print(f"    Deleted CSV: {vd.name}")
                else:
                    print(f"    No CSV (will run): {vd.name}")
            cmd = [python, str(script_dir / '03_batch_slam.py'),
                   '--input_dir',       str(demo_dir),
                   '--map_path',        str(map_path),
                   '--max_lost_frames', str(max_lost_frames2),
                   '--num_workers',     str(num_workers)]
            if local:
                cmd.append('--local')
            elif no_docker_pull:
                cmd.append('--no_docker_pull')
            run(cmd, '03_batch_slam pass2')
            n_ok_p2, _ = slam_summary(demo_dir, f"pass 2, -ml {max_lost_frames2}")
            rescued = n_ok_p2 - n_ok_p1
            if rescued > 0:
                ok(f"Pass 2 rescued {rescued} additional demo(s)")
            else:
                warn("Pass 2 did not rescue any additional demos (likely map coverage gaps)")
        else:
            if not failed_after_p1:
                ok("All demos passed on pass 1 — skipping pass 2")
            else:
                warn(f"max_lost_frames2 ({max_lost_frames2}) <= max_lost_frames ({max_lost_frames}) — skipping pass 2")

        # ── STEP 04 ──────────────────────────────────────────────
        step_header(4, '04_detect_aruco')
        cmd = [python, str(script_dir / '04_detect_aruco.py'),
               '--input_dir',         str(demo_dir),
               '--camera_intrinsics', str(camera_intrinsics),
               '--aruco_yaml',        str(aruco_config),
               '--num_workers',       str(num_workers)]
        if redo_aruco:
            cmd.append('--redo')
        run(cmd, '04_detect_aruco')

        # ── STEP 05 ──────────────────────────────────────────────
        step_header(5, '05_run_calibrations')
        result = run([python, str(script_dir / '05_run_calibrations.py'), str(session)], '05_run_calibrations')
        slam_tag = mapping_dir / 'tx_slam_tag.json'
        if slam_tag.is_file():
            ok("tx_slam_tag.json generated")
        else:
            warn("tx_slam_tag.json not found — table marker (ID 13) may not have been visible in mapping video")
        gripper_ranges = list(demo_dir.glob('gripper_calibration_*/gripper_range.json'))
        print(f"  Gripper ranges   : {len(gripper_ranges)} calibrated")

        # ── STEP 06 ──────────────────────────────────────────────
        step_header(6, '06_generate_dataset_plan')
        result = run([python, str(script_dir / '06_generate_dataset_plan.py'), '--input', str(session)], '06_generate_dataset_plan')
        n_episodes = check_dataset_plan(plan_path)
        if n_episodes == 0:
            fail("No episodes generated — aborting before zarr generation.")
            continue

        # ── STEP 07 ──────────────────────────────────────────────
        step_header(7, '07_generate_replay_buffer')
        cmd = [python, str(script_dir / '07_generate_replay_buffer.py'),
               '--output',      str(zarr_out),
               '--num_workers', str(num_workers),
               str(session)]
        result = run(cmd, '07_generate_replay_buffer')
        zarr_out = pathlib.Path(zarr_out)
        if zarr_out.exists():
            size_mb = sum(f.stat().st_size for f in zarr_out.rglob('*') if f.is_file()) / 1e6 \
                if zarr_out.is_dir() else zarr_out.stat().st_size / 1e6
            ok(f"replay_buffer.zarr saved ({size_mb:.0f} MB)")
        else:
            fail("replay_buffer.zarr not found after step 07")

        # ── FINAL SUMMARY ─────────────────────────────────────────
        print(f"\n{'='*50}")
        print(f"  PIPELINE COMPLETE : {session.name}")
        print(f"  Episodes          : {n_episodes}")
        print(f"  Zarr              : {zarr_out}")
        print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
