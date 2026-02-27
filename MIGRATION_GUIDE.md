# UMI Migration Guide: UR5/WSG → UR10/Dynamixel XH540/Hero 12 Black

This document summarizes all changes made to adapt the Stanford UMI repository for:
- **Robot**: UR5/UR5e → UR10
- **Gripper**: WSG Gripper → Dynamixel XH540 Gripper
- **Camera**: Hero 9 → Hero 12 Black

## Summary of Changes

### 1. Robot Controller Updates

#### Files Modified:
- `umi/real_world/umi_env.py`
- `umi/real_world/bimanual_umi_env.py`
- `umi/real_world/real_env.py`
- `diffusion_policy/real_world/umi_env.py`
- `diffusion_policy/real_world/real_env.py`
- `scripts_real/eval_real_umi.py`
- `scripts_real/replay_real_bimanual_umi.py`
- `scripts_real/eval_real_bimanual_umi.py`
- `scripts_real/demo_real_bimanual_robots.py`
- `example/eval_robots_config.yaml`

#### Changes:
- **Robot type**: Changed from `'ur5'`/`'ur5e'` to `'ur10'`
- **Frequency**: Updated comments from "UR5 CB3 RTDE" to "UR10 RTDE" (frequency remains 500Hz)
- **Robot type checks**: Updated `robot_type.startswith('ur5')` to `robot_type.startswith('ur10')`
- **Joint initialization**: No changes needed (UR5 and UR10 use same joint initialization)

### 2. Gripper Controller Updates

#### New File Created:
- `umi/real_world/dynamixel_xh540_controller.py` - New Dynamixel XH540 controller matching WSGController interface

#### Files Modified:
- `umi/real_world/umi_env.py` - Replaced WSGController with DynamixelXH540Controller
- `umi/real_world/bimanual_umi_env.py` - Replaced WSGController with DynamixelXH540Controller
- `umi/real_world/real_env.py` - Replaced WSGController with DynamixelXH540Controller
- `example/eval_robots_config.yaml` - Updated gripper configuration format
- `scripts_real/replay_real_bimanual_umi.py` - Updated gripper config
- `scripts_real/eval_real_bimanual_umi.py` - Updated gripper config
- `scripts_real/demo_real_bimanual_robots.py` - Updated gripper config

#### Gripper Configuration Changes:
**Old WSG Format:**
```yaml
grippers:
  - gripper_ip: "192.168.0.18"
    gripper_port: 1000
    gripper_obs_latency: 0.01
    gripper_action_latency: 0.1
```

**New Dynamixel Format:**
```yaml
grippers:
  - gripper_port: "/dev/ttyUSB0"  # Serial port path
    gripper_baudrate: 1000000      # Dynamixel baudrate
    dynamixel_id: 1               # Dynamixel servo ID
    gripper_obs_latency: 0.01
    gripper_action_latency: 0.1
```

### 3. Camera Updates

#### Files Modified:
- `scripts_slam_pipeline/06_generate_dataset_plan.py` - Updated comment to include Hero 12

#### Changes:
- Camera offset constant comment updated to include Hero 12 (offset value remains the same: 0.01465m)
- Hero 12 Black uses the same mounting offset as Hero 9/10/11

## Parameters to Configure

### 1. Robot Configuration (`example/eval_robots_config.yaml`)

```yaml
robots:
  - robot_type: "ur10"  # Changed from ur5e
    robot_ip: "YOUR_UR10_IP"  # Update with your UR10 IP address
    robot_obs_latency: 0.0001
    robot_action_latency: 0.1
    tcp_offset: 0.235  # Adjust based on your end-effector
    height_threshold: -0.024  # Adjust for your table height
    sphere_radius: 0.1
    sphere_center: [0, -0.06, -0.185]
```

### 2. Gripper Configuration (`example/eval_robots_config.yaml`)

```yaml
grippers:
  - gripper_port: "/dev/ttyUSB0"  # Serial port - CHECK YOUR SYSTEM
    gripper_baudrate: 1000000     # Default Dynamixel baudrate
    dynamixel_id: 1               # Your Dynamixel servo ID
    gripper_obs_latency: 0.01
    gripper_action_latency: 0.1
```

**Important Notes:**
- Find your serial port: `ls /dev/ttyUSB*` or `ls /dev/ttyACM*` on Linux
- Verify Dynamixel ID using Dynamixel Wizard or similar tool
- Adjust baudrate if your Dynamixel is configured differently

### 3. Dynamixel XH540 Controller Setup

The new `DynamixelXH540Controller` requires:
- **dynamixel_sdk**: Install using `pip install dynamixel-sdk`
- Or implement custom driver if using different library

**Controller Parameters:**
- `port`: Serial port path (e.g., `/dev/ttyUSB0`)
- `baudrate`: Communication baudrate (default: 1000000)
- `dynamixel_id`: Servo ID (default: 1)
- `min_position`: Minimum gripper position (default: 0)
- `max_position`: Maximum gripper position (default: 4095 for 12-bit XH540)
- `use_meters`: If True, converts position to meters (assumes 110mm max opening)

### 4. Camera Configuration

Hero 12 Black should work with existing camera setup. The camera offset constant (0.01465m) applies to Hero 9, 10, 11, and 12.

## How to Launch Features

### 1. Single Arm Setup

**Demo/Recording:**
```bash
python scripts_real/demo_real_umi.py \
    --output ./data/demo \
    --robot_ip YOUR_UR10_IP \
    --gripper_ip /dev/ttyUSB0  # Note: now serial port, not IP
```

**Evaluation:**
```bash
python scripts_real/eval_real_umi.py \
    --input ./checkpoints/policy.ckpt \
    --output ./data/eval \
    --robot_ip YOUR_UR10_IP \
    --gripper_ip /dev/ttyUSB0 \
    --robot_type ur10
```

### 2. Bimanual Setup

**Using Config File:**
```bash
python scripts_real/eval_real_bimanual_umi.py \
    --input ./checkpoints/policy.ckpt \
    --output ./data/eval \
    --config example/eval_robots_config.yaml
```

**Direct Configuration:**
Update the `robots_config` and `grippers_config` dictionaries in:
- `scripts_real/demo_real_bimanual_robots.py`
- `scripts_real/eval_real_bimanual_umi.py`
- `scripts_real/replay_real_bimanual_umi.py`

### 3. Training Pipeline

The training pipeline (`train.py`, `eval_real.py`) should work with updated configurations. Ensure:
- Robot IP addresses are correct
- Gripper serial ports are accessible
- Camera devices are properly connected

## Important Notes

### Dynamixel Controller Implementation

The `DynamixelXH540Controller` includes:
- Full interface matching `WSGController` for compatibility
- Support for dynamixel_sdk (official SDK)
- Placeholder mode if SDK not available (for testing)

**TODO for Full Implementation:**
1. Install dynamixel_sdk: `pip install dynamixel-sdk`
2. Verify XH540 control table addresses match implementation
3. Test and calibrate position scaling (`use_meters` parameter)
4. Adjust `min_position` and `max_position` based on your gripper's actual range

### Robot Type Compatibility

The code now checks for `robot_type.startswith('ur10')`. If you need to support both UR5 and UR10:
- Add conditional logic: `if robot_type.startswith('ur5') or robot_type.startswith('ur10')`
- Adjust frequency based on robot type (UR5 CB2: 125Hz, UR5e/UR10: 500Hz)

### Script Compatibility

Some scripts still use `gripper_ip` as a command-line argument. These will need manual updates:
- `scripts_real/demo_real_umi.py`
- `scripts_real/eval_real_umi.py`
- `scripts_real/demo_real_robot.py`
- `scripts_real/eval_real_robot.py`

Consider updating these to use `--gripper_port` instead of `--gripper_ip`.

## Testing Checklist

- [ ] Verify UR10 connection and control
- [ ] Test Dynamixel XH540 communication
- [ ] Calibrate gripper position scaling
- [ ] Verify camera capture with Hero 12 Black
- [ ] Test single arm demo/recording
- [ ] Test bimanual setup (if applicable)
- [ ] Verify policy evaluation pipeline
- [ ] Test training data collection

## Troubleshooting

### Dynamixel Connection Issues
- Check serial port permissions: `sudo chmod 666 /dev/ttyUSB0`
- Verify baudrate matches Dynamixel configuration
- Check Dynamixel ID using Dynamixel Wizard

### Robot Connection Issues
- Verify UR10 IP address and network connectivity
- Check RTDE port (default: 30004)
- Ensure robot is in remote control mode

### Camera Issues
- Hero 12 Black should work identically to Hero 9
- Verify capture card detection: `ls /dev/video*`
- Check camera resolution settings match Hero 12 capabilities
