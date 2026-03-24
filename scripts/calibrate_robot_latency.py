# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
print(ROOT_DIR)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import click
import time
import numpy as np
from multiprocessing.managers import SharedMemoryManager
from umi.real_world.rtde_interpolation_controller import RTDEInterpolationController
from umi.common.precise_sleep import precise_wait
from umi.common.latency_util import get_latency
from matplotlib import pyplot as plt

# %%
@click.command()
@click.option('-rh', '--robot_hostname', default='172.16.0.3')
@click.option('-f', '--frequency', type=float, default=30)
@click.option('-a', '--amplitude', type=float, default=0.02, help='Sine amplitude in meters (default: 2cm)')
@click.option('-d', '--duration', type=float, default=10.0, help='Duration in seconds')
def main(robot_hostname, frequency, amplitude, duration):
    max_pos_speed = 0.5
    max_rot_speed = 1.2
    cube_diag = np.linalg.norm([1,1,1])
    tcp_offset = 0.284
    dt = 1/frequency

    with SharedMemoryManager() as shm_manager:
        with RTDEInterpolationController(
            shm_manager=shm_manager,
            robot_ip=robot_hostname,
            frequency=500,
            lookahead_time=0.1,
            gain=300,
            max_pos_speed=max_pos_speed*cube_diag,
            max_rot_speed=max_rot_speed*cube_diag,
            tcp_offset_pose=[0,0,tcp_offset,0,0,0],
            get_max_k=int(duration * 500 * 1.5),
            verbose=False
        ) as controller:
            print('Ready! Starting automated sine wave motion...')
            print(f'Amplitude: {amplitude}m, Duration: {duration}s')
            print('Press Ctrl+C to abort.')

            state = controller.get_state()
            start_pose = state['ActualTCPPose'].copy()
            print(f'Start pose: {np.array2string(start_pose, precision=4)}')

            n_steps = int(duration / dt)
            t_start = time.time() + 1.0  # 1s delay to settle

            t_target = []
            x_target = []

            for i in range(n_steps):
                t_command = t_start + (i + 1) * dt
                t_now = i * dt

                target_pose = start_pose.copy()
                # Sine wave on X axis (different freq per axis for better cross-corr)
                target_pose[0] += amplitude * np.sin(2 * np.pi * 0.5 * t_now)
                # Sine wave on Y axis at different frequency
                target_pose[1] += amplitude * np.sin(2 * np.pi * 0.7 * t_now)
                # Sine wave on Z axis at different frequency (smaller amplitude for safety)
                target_pose[2] += (amplitude * 0.5) * np.sin(2 * np.pi * 0.3 * t_now)

                t_target.append(t_command)
                x_target.append(target_pose.copy())

                controller.schedule_waypoint(target_pose, t_command)
                precise_wait(t_command - 0.5, time_func=time.time)

            # Wait for motion to complete
            print('Waypoints scheduled. Waiting for motion to complete...')
            precise_wait(t_start + duration + 1.0, time_func=time.time)

            states = controller.get_all_state()

    t_target = np.array(t_target)
    x_target = np.array(x_target)
    t_actual = states['robot_receive_timestamp']
    x_actual = states['ActualTCPPose']

    dim_names = ['X', 'Y', 'Z', 'Rx', 'Ry', 'Rz']
    # Only plot X, Y, Z (the dims we moved)
    plot_dims = [0, 1, 2]
    n_plots = len(plot_dims)
    fig, axes = plt.subplots(n_plots, 3)
    fig.set_size_inches(15, 5 * n_plots, forward=True)

    latencies = []
    for row_idx, dim_idx in enumerate(plot_dims):
        latency, info = get_latency(x_target[..., dim_idx], t_target,
                                     x_actual[..., dim_idx], t_actual)
        latencies.append(latency)
        print(f'{dim_names[dim_idx]}: latency = {latency:.4f}s')

        row = axes[row_idx]
        ax = row[0]
        ax.plot(info['lags'], info['correlation'])
        ax.set_xlabel('lag (s)')
        ax.set_ylabel('cross-correlation')
        ax.set_title(f"{dim_names[dim_idx]} Cross Correlation")

        ax = row[1]
        ax.plot(t_target, x_target[..., dim_idx], label='target')
        ax.plot(t_actual, x_actual[..., dim_idx], label='actual')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('position (m)')
        ax.legend()
        ax.set_title(f"{dim_names[dim_idx]} Raw observation")

        ax = row[2]
        t_samples = info['t_samples'] - info['t_samples'][0]
        ax.plot(t_samples, info['x_target'], label='target')
        ax.plot(t_samples - latency, info['x_actual'], label='actual-latency')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('position (normalized)')
        ax.legend()
        ax.set_title(f"{dim_names[dim_idx]} Aligned (latency={latency:.4f}s)")

    avg_latency = np.mean(latencies)
    print(f'\n=== Average robot action latency: {avg_latency:.4f}s ===')
    print(f'Update this value in eval_real.py / eval_robots_config.yaml')

    fig.suptitle(f'Robot Action Latency (avg={avg_latency:.4f}s)', fontsize=14)
    fig.tight_layout()
    plt.savefig('robot_latency_calibration.png', dpi=150)
    print('Plot saved to robot_latency_calibration.png')
    plt.show()

# %%
if __name__ == '__main__':
    main()
