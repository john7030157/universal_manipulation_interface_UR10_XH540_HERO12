## UMI on UR10 + Dynamixel XH540 + Hero 12 (Ubuntu 22.04)

This README documents how to use this fork of the **Universal Manipulation Interface (UMI)** with:

- **Robot**: UR10
- **Gripper**: Custom Dynamixel XH540-based parallel jaw gripper
- **Camera**: GoPro Hero 12 Black via HDMI capture card
- **OS**: Ubuntu 22.04

The high-level ideas and capabilities follow the official UMI project page [`umi-gripper.github.io`](https://umi-gripper.github.io/).

For general background, SLAM pipeline, and training details, also see the original `README.md` in this repo.

---

## 1. System Setup on Ubuntu 22.04

### 1.1. Base software (same as upstream UMI)

1. **Install Docker** (if you plan to use the provided SLAM Docker image)  
   Follow Docker’s official docs for Ubuntu and complete the post-install steps so you can run Docker without `sudo`.

2. **Install system-level dependencies**:

   ```bash
   sudo apt update
   sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf libspnav-dev spacenavd
   ```

3. **Install Miniforge / Conda environment**:

   ```bash
   # From repo root
   mamba env create -f conda_environment.yaml
   conda activate umi
   ```

4. **(Optional) SLAM Docker**:  
   The original repo uses an ORB-SLAM3 Docker image. See the upstream `README.md` and the linked SLAM Docker docs if you need full SLAM from raw human videos.

5. **Install Dynamixel SDK** (for XH540 gripper):

   ```bash
   conda activate umi
   pip install dynamixel-sdk
   ```

---

## 2. Hardware Overview & Wiring

This fork assumes the same *conceptual* hardware design as the original UMI system [`umi-gripper.github.io`](https://umi-gripper.github.io/), but with different robot and gripper:

- **UR10** robot arm
- **Custom Dynamixel XH540 parallel jaw gripper** mounted at the UR10 flange
- **GoPro Hero 12 Black** rigidly attached near the gripper, feeding HDMI into a capture card
- **(Optional) 3Dconnexion SpaceMouse** for teleoperation and safety

You should follow the upstream **Hardware Guide** for mechanical layout, then adapt:

- Replace the UR5e flange adapter with your UR10-compatible mount
- Replace the WSG gripper with your Dynamixel XH540 gripper and mount the Hero 12 in a similar pose

---

## 3. UR10 Configuration (Payload, CoG, TCP)

The original UMI instructions assume a WSG50 gripper on UR5e with a known payload:

- **Mass**: 1.81 kg  
- **CoG**: (2, -6, 37) mm in the UR base TCP config (from original `README.md`)

Since you now have a **custom Dynamixel gripper**, you **must re-calibrate**:

### 3.1. Payload mass & CoG calibration (UR teach pendant)

1. **Remove extra objects** (only UR10 + gripper attached, no tools or cups).
2. On the UR teach pendant:
   - Go to **Installation → Payload**.
   - Use the UR’s built-in **payload estimation** wizard:
     - Follow prompts to move the arm through required poses.
     - The wizard will estimate **mass** and **center of gravity**.
3. Once you get stable values:
   - Set **Mass** to the estimated value.
   - Set **CoG (CX/CY/CZ)** to the estimated values in **mm**.
4. Save the installation.

These values are used by the **UR controller itself** and by the RTDE controller for better dynamic performance. You do *not* need to hard-code them into Python; UR will use them for gravity compensation and motion planning.

### 3.2. TCP / tool center point

The code sets TCP offsets in Python via `tcp_offset` (e.g. 0.235 m) and passes them to `RTDEInterpolationController`. You should:

1. Measure or CAD-compute **distance from UR10 flange to the gripper fingertip center** (where you want the policy’s “EEF pose” to reside).
2. Update:
   - `tcp_offset` / `tcp_offset_pose` in:
     - `example/eval_robots_config.yaml` (`tcp_offset`)
     - Any scripts where `tcp_offset` is hard-coded (search for `tcp_offset=`).
3. For small deviations (~1–2 cm), policies are usually robust; still, accurate TCP helps especially for tight tasks (cup placement, cloth folding).

---

## 4. Dynamixel XH540 Gripper Configuration & Calibration

The original UMI stack used a WSG gripper over Ethernet with a custom binary protocol. This fork replaces it with:

- `umi/real_world/dynamixel_xh540_controller.py`  
  A process-based interpolation controller that **matches the WSGController interface** but talks to a Dynamixel XH540 over serial via `dynamixel-sdk`.

### 4.1. Physical & OS setup

1. Connect your Dynamixel bus to the PC via a U2D2 or similar USB–TTL interface.
2. On Ubuntu:

   ```bash
   ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null
   # Pick the port, e.g. /dev/ttyUSB0
   sudo chmod 666 /dev/ttyUSB0   # or add a udev rule for persistence
   ```

3. Use **Dynamixel Wizard / U2D2** to:
   - Confirm the **ID** of your XH540 (e.g. `1`)
   - Confirm **baudrate** (default often `1000000`)
   - Confirm protocol (XH540 uses **Protocol 2.0**)

### 4.2. Gripper range calibration (stroke)

UMI expects a **normalized gripper width** signal (`robotX_gripper_width` in meters). With a custom gripper, you must:

1. Determine **physical stroke**:
   - Fully **open** the gripper → measure jaw-to-jaw distance (e.g. 0.11 m).
   - Fully **close** the gripper → measure jaw-to-jaw distance (e.g. 0.0–0.01 m).
2. In `DynamixelXH540Controller`:
   - `min_position`: encoder value corresponding to **closed**.
   - `max_position`: encoder value corresponding to **open**.
3. Map encoder → meters:
   - The controller currently assumes ~0.11m max opening:
     - `scale = (max_position - min_position) / 0.11` when `use_meters=True`.
   - If your max stroke is different, adjust this constant.
4. Verify by logging:
   - Start a simple script that uses `DynamixelXH540Controller` and prints:
     - `gripper_position`
   - Manually command open/close and ensure values in meters match reality.

> If you want a more formal procedure, you can adapt `scripts/calibrate_gripper_range.py`:
> - Replace its `WSGController` usage with `DynamixelXH540Controller`.
> - Programmatically sweep the gripper between open/close and fit a linear mapping.

### 4.3. Gripper mass & CoG

Your **Dynamixel gripper mass & CoG** are covered by the UR payload wizard (Section 3.1). No additional parameters need to be set in the Python controller unless you want precise torque-to-force mapping.

---

## 5. GoPro Hero 12 Black Setup

UMI’s design is camera-centric, with a wrist-mounted GoPro providing **all** visual context for policy learning [`umi-gripper.github.io`](https://umi-gripper.github.io/). For Hero 12 Black:

1. Install **GoPro Labs firmware** (as recommended upstream).
2. Set date and time.
3. Use the **clean HDMI output QR code** provided in `assets/QR-MHDMI1mV0r27Tp60fWe0hS0sLcFg1dV.png` (same as Hero 9/10/11).
4. Connect the GoPro to an HDMI capture card (e.g., Elgato CamLink 4K).
5. On Ubuntu, verify the capture device:

   ```bash
   ls /dev/video*
   ```

6. The UMI codebase automatically discovers and resets Elgato devices via:
   - `umi/common/usb_util.py`
   - `umi/real_world/uvc_camera.py` and `multi_uvc_camera.py`

No additional changes are needed specifically for Hero 12; it behaves the same as Hero 9/10/11 for UMI’s purposes.

---

## 6. Configuration Files to Edit

### 6.1. `example/eval_robots_config.yaml`

This repo’s copy has been updated for UR10 + Dynamixel, but you must enter your values:

```yaml
{
  "robots": [
    {
      "robot_type": "ur10",
      "robot_ip": "192.168.0.8",           # ← your UR10 IP
      "robot_obs_latency": 0.0001,
      "robot_action_latency": 0.1,
      "tcp_offset": 0.235,                 # ← adjust for your tool length
      "height_threshold": -0.024,          # table height for collision avoidance
      "sphere_radius": 0.1,
      "sphere_center": [0, -0.06, -0.185]
    }
  ],

  "grippers": [
    {
      "gripper_port": "/dev/ttyUSB0",      # ← your Dynamixel port
      "gripper_baudrate": 1000000,
      "dynamixel_id": 1,
      "gripper_obs_latency": 0.01,
      "gripper_action_latency": 0.1
    }
  ],

  "tx_left_right": [
    # Only needed for bimanual; see original file
  ]
}
```

For **bimanual**, add a second robot and gripper entry with the second IP and Dynamixel port/ID.

### 6.2. Hard-coded configs in `scripts_real/`

Several scripts have in-file `robots_config` and `grippers_config` dictionaries (for quick testing). Search for:

- `robots_config = [`
- `grippers_config = [`

Ensure:

- `robot_type: 'ur10'`
- Correct `robot_ip` values
- `gripper_port`, `gripper_baudrate`, and `dynamixel_id` match your hardware

---

## 7. Running All Main Features

This section summarizes end-to-end usage (Ubuntu 22.04, UR10, Dynamixel).

### 7.1. SLAM pipeline from human handheld videos

Purpose: from **UMI gripper handheld demonstrations** (human using the handheld gripper with GoPro), produce SLAM trajectories and a replay buffer for training.

1. Prepare a folder of raw videos as described in the original Data Collection Instruction.
2. Run SLAM:

   ```bash
   conda activate umi
   python run_slam_pipeline.py YOUR_SESSION_DIR
   ```

3. Generate replay buffer:

   ```bash
   python scripts_slam_pipeline/07_generate_replay_buffer.py \
       -o YOUR_SESSION_DIR/dataset.zarr.zip \
       YOUR_SESSION_DIR
   ```

This produces a `.zarr.zip` dataset compatible with the diffusion policy training code.

### 7.2. Training Diffusion Policies

Single-GPU example (matching upstream UMI training setup):

```bash
conda activate umi
python train.py \
  --config-name=train_diffusion_unet_timm_umi_workspace \
  task.dataset_path=YOUR_SESSION_DIR/dataset.zarr.zip
```

You can also use the **official in-the-wild cup dataset** from the UMI project page and train policies that you then deploy on your UR10.

### 7.3. Real-world deployment (single-arm UR10 + Dynamixel)

There are two main layers:

1. **Low-level async envs**: `umi/real_world/umi_env.py` and `umi/real_world/real_env.py`
   - They manage:
     - **UR10 RTDE** via `RTDEInterpolationController`
     - **Dynamixel XH540 gripper** via `DynamixelXH540Controller`
     - **GoPro cameras** via `MultiUvcCamera`
2. **High-level scripts**: call these envs, handle teleoperation, logging, and policy rollout.

#### 7.3.1. Cup arrangement policy (UR10)

1. Download/pretrain or place your policy checkpoint, e.g.:

   ```bash
   wget https://real.stanford.edu/umi/data/pretrained_models/cup_wild_vit_l_1img.ckpt
   ```

2. Grant USB permissions for the HDMI capture card:

   ```bash
   sudo chmod -R 777 /dev/bus/usb
   ```

3. Update:
   - `example/eval_robots_config.yaml` for UR10 + Dynamixel (Section 6.1).

4. Run evaluation:

   ```bash
   conda activate umi
   python eval_real.py \
     --robot_config=example/eval_robots_config.yaml \
     -i cup_wild_vit_l_1img.ckpt \
     -o data/eval_cup_wild_ur10_dynamixel
   ```

5. Use the SpaceMouse to teleop:
   - Buttons: open/close gripper, deadman switch
   - `C` key: start policy rollout
   - `S` key: stop policy

From the policy’s perspective, this UR10 + Dynamixel setup is just another robot with a parallel jaw stroke ≥ 85mm, as described in the UMI project page.

### 7.4. Bimanual deployment

For bimanual tasks (e.g., cloth folding):

1. Ensure each UR10 has:
   - Unique IP (`robot_ip`)
   - Calibration and payload set correctly
2. Ensure each Dynamixel gripper has:
   - Unique `dynamixel_id`
   - Unique serial port (`/dev/ttyUSB0`, `/dev/ttyUSB1`, …)
3. Update:
   - `example/eval_robots_config.yaml` with 2 robots and 2 grippers
   - `scripts_real/demo_real_bimanual_robots.py`
   - `scripts_real/eval_real_bimanual_umi.py`
   - `scripts_real/replay_real_bimanual_umi.py`
4. Run, e.g.:

   ```bash
   conda activate umi
   python scripts_real/demo_real_bimanual_robots.py \
     --output data/demo_bimanual \
     --robot_ip 192.168.0.8 \
     --gripper_ip /dev/ttyUSB0
   ```

The bimanual env `umi/real_world/bimanual_umi_env.py` now constructs:

- 2× `RTDEInterpolationController` for UR10 arms
- 2× `DynamixelXH540Controller` for the XH540 grippers

---

## 8. Calibration Processes (Summary)

### 8.1. Robot-side calibration

- **Payload mass & CoG**: UR10 pendant wizard (Section 3.1).
- **TCP / tool length**: adjust `tcp_offset` in config and scripts based on CAD or measurement.

### 8.2. Gripper-side calibration

- **Stroke mapping (encoder → meters)**:
  - Measure physical open/close distances.
  - Set `min_position`, `max_position` and scaling in `DynamixelXH540Controller`.
  - Optionally, adapt `scripts/calibrate_gripper_range.py` for automatic logging and interpolation.

- **Latency calibration**:
  - If needed, you can adapt `scripts/calibrate_gripper_latency.py` to use the Dynamixel controller and confirm `gripper_obs_latency` and `gripper_action_latency` parameters.

These calibrations ensure that the **UMI policy interface** (EEF pose + gripper width) remains physically meaningful on your new hardware, which is central to UMI’s hardware-agnostic claim.

---

## 9. Simulation Availability

You asked whether a **simulation is available** before testing on the real UR10 + Dynamixel setup.

### 9.1. What’s in this repo

This repository already includes several **simulation environments** for Diffusion Policy:

- `diffusion_policy/env/block_pushing/` – PyBullet block pushing task
- `diffusion_policy/env/kitchen/` – Kitchen manipulation tasks (Franka-based)
- `diffusion_policy/env/pusht/` – 2D Push-T environment

These are great for **testing training scripts, model architectures, and general workflows**, but:

> They are **not a direct simulation of the UR10 + wrist-mounted GoPro + Dynamixel gripper UMI stack**.

The real-world UMI part (`umi/real_world/…`) assumes physical hardware.

### 9.2. Options for UR10 + Dynamixel simulation

There is **no out-of-the-box UR10+UMI simulation** in this repo, but you have a few options:

1. **URSim + ROS / Gazebo / Isaac Sim integration (recommended)**:
   - Use UR’s official URSim (or ROS drivers) to simulate UR10.
   - Use Gazebo / Isaac Sim / Mujoco to:
     - Spawn a UR10 with a simple parallel jaw gripper.
     - Mount a virtual camera at the wrist (matching the GoPro pose).
   - Implement a **simulation backend** that exposes the same interface as `RTDEInterpolationController` + `DynamixelXH540Controller`:
     - Same `schedule_waypoint` / `get_all_state` APIs.
   - Wire this into a “sim” version of `umi_env` so Diffusion Policy sees the same observation/action spaces.

2. **Use existing DP envs for algorithm debugging**:
   - Before working with real hardware, you can:
     - Train and evaluate on `block_pushing` and `kitchen` envs.
     - Confirm training, logging, and inference pipelines behave as expected.
   - Then switch to the real-world envs once you’re confident in the pipeline.

3. **Offline replay “simulation”**:
   - Another lightweight alternative is to use recorded **real-world episodes**:
     - Run policies in a “replay-only” mode where the robot/gripper are not commanded, but observations are drawn from logs.
     - This lets you test **policy code and timing** without moving hardware (but not control dynamics).

### 9.3. Practical answer

- **There is no ready-made UR10+UMI simulator in this repo.**  
- However:
  - You **can** safely debug most of the ML and data pipelines using the included simulated tasks (block pushing, kitchen, pusht).
  - You **can** build your own UR10 + camera + gripper simulation layer that emulates `RTDEInterpolationController`/`DynamixelXH540Controller`, then plug it into the same UMI env interface.

---

## 10. Recommended Bring-up Order

1. **Set up Conda environment and run simulated tasks**  
   - Confirm Diffusion Policy training and evaluation works on `block_pushing`.
2. **Verify GoPro + capture card**  
   - Use `ffplay` or a simple OpenCV script to confirm `/dev/video*` works.
3. **Verify Dynamixel communication**  
   - Use a small script with `dynamixel-sdk` to open/close the gripper.
4. **Run UMI real envs without policies**  
   - Use SpaceMouse teleop scripts to:
     - Move UR10 in Cartesian space.
     - Open/close the gripper from Python (no ML policy yet).
5. **Calibrate stroke + payload**  
   - Update `DynamixelXH540Controller` scaling.
   - Run UR10 payload/CoG estimation on the pendant.
6. **Deploy a simple policy (e.g., cup arrangement)**  
   - Start from a pretrained checkpoint or a simple trained policy.

Following this order keeps risk low and makes debugging much easier.

