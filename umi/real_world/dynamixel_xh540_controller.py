import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from umi.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.precise_sleep import precise_wait
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator


class Command(enum.Enum):
    SHUTDOWN = 0
    SCHEDULE_WAYPOINT = 1
    RESTART_PUT = 2


class DynamixelXH540Controller(mp.Process):
    """
    Controller for Dynamixel XH540-based gripper.
    This follows the same interface as WSGController for compatibility.
    """
    def __init__(self,
            shm_manager: SharedMemoryManager,
            port='/dev/ttyUSB0',  # Serial port for Dynamixel
            baudrate=1000000,  # Default Dynamixel baudrate
            dynamixel_id=1,  # Dynamixel servo ID
            frequency=30,
            home_to_open=True,
            move_max_speed=200.0,  # Max speed in position units per second
            get_max_k=None,
            command_queue_size=1024,
            launch_timeout=3,
            receive_latency=0.0,
            use_meters=False,
            min_position=0,  # Minimum gripper position (closed)
            max_position=4095,  # Maximum gripper position (open) - XH540 has 12-bit resolution
            verbose=False
            ):
        super().__init__(name="DynamixelXH540Controller")
        self.port = port
        self.baudrate = baudrate
        self.dynamixel_id = dynamixel_id
        self.frequency = frequency
        self.home_to_open = home_to_open
        self.move_max_speed = move_max_speed
        self.launch_timeout = launch_timeout
        self.receive_latency = receive_latency
        self.min_position = min_position
        self.max_position = max_position
        self.verbose = verbose
        
        # Scale factor: if use_meters=True, convert from meters to position units
        # Assuming max gripper opening is ~0.11m (110mm), scale accordingly
        self.scale = (max_position - min_position) / 0.11 if use_meters else 1.0

        if get_max_k is None:
            get_max_k = int(frequency * 10)
        
        # build input queue
        example = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=command_queue_size
        )
        
        # build ring buffer
        example = {
            'gripper_state': 0,
            'gripper_position': 0.0,
            'gripper_velocity': 0.0,
            'gripper_force': 0.0,
            'gripper_measure_timestamp': time.time(),
            'gripper_receive_timestamp': time.time(),
            'gripper_timestamp': time.time()
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )
        
        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[DynamixelXH540Controller] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.SHUTDOWN.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()
    
    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    # ========= command methods ============
    def schedule_waypoint(self, pos: float, target_time: float):
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': pos,
            'target_time': target_time
        }
        self.input_queue.put(message)

    def restart_put(self, start_time):
        self.input_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'target_time': start_time
        })
    
    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    
    # ========= helper methods for Dynamixel communication ============
    def _read_position(self, dynamixel_interface):
        """
        Read current position from Dynamixel.
        Returns position in raw units (0-4095 for XH540).
        """
        # TODO: Implement actual Dynamixel read using dynamixel_sdk or pydxl
        # Example using dynamixel_sdk:
        # position, result, error = dynamixel_interface.read(self.dynamixel_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
        # return position if result == COMM_SUCCESS else None
        raise NotImplementedError("Implement Dynamixel position reading")
    
    def _read_velocity(self, dynamixel_interface):
        """
        Read current velocity from Dynamixel.
        """
        # TODO: Implement actual Dynamixel velocity reading
        raise NotImplementedError("Implement Dynamixel velocity reading")
    
    def _read_load(self, dynamixel_interface):
        """
        Read current load/torque from Dynamixel.
        """
        # TODO: Implement actual Dynamixel load reading
        raise NotImplementedError("Implement Dynamixel load reading")
    
    def _write_position(self, dynamixel_interface, position, velocity=0):
        """
        Write target position to Dynamixel.
        """
        # TODO: Implement actual Dynamixel write using dynamixel_sdk or pydxl
        # Example:
        # dynamixel_interface.write(self.dynamixel_id, ADDR_GOAL_POSITION, position)
        # dynamixel_interface.write(self.dynamixel_id, ADDR_PROFILE_VELOCITY, velocity)
        raise NotImplementedError("Implement Dynamixel position writing")
    
    def _enable_torque(self, dynamixel_interface, enable=True):
        """
        Enable/disable torque on Dynamixel.
        """
        # TODO: Implement torque enable/disable
        raise NotImplementedError("Implement Dynamixel torque control")
    
    # ========= main loop in process ============
    def run(self):
        # Import Dynamixel SDK here to avoid import issues in main process
        try:
            # Try importing dynamixel_sdk (official SDK)
            try:
                from dynamixel_sdk import PortHandler, PacketHandler
                USE_DYNAMIXEL_SDK = True
            except ImportError:
                USE_DYNAMIXEL_SDK = False
                if self.verbose:
                    print("[DynamixelXH540Controller] dynamixel_sdk not found, using placeholder")
        except Exception as e:
            USE_DYNAMIXEL_SDK = False
            if self.verbose:
                print(f"[DynamixelXH540Controller] Error importing SDK: {e}")
        
        # Initialize Dynamixel interface
        try:
            if USE_DYNAMIXEL_SDK:
                # Initialize port handler and packet handler
                port_handler = PortHandler(self.port)
                packet_handler = PacketHandler(2.0)  # Protocol 2.0 for XH series
                
                # Open port
                if port_handler.openPort():
                    if self.verbose:
                        print(f"[DynamixelXH540Controller] Port opened: {self.port}")
                else:
                    raise Exception(f"Failed to open port: {self.port}")
                
                # Set baudrate
                if not port_handler.setBaudRate(self.baudrate):
                    raise Exception(f"Failed to set baudrate: {self.baudrate}")
                
                # Enable torque
                ADDR_TORQUE_ENABLE = 64  # XH540 address for torque enable
                TORQUE_ENABLE = 1
                dxl_comm_result, dxl_error = packet_handler.write1ByteTxRx(
                    port_handler, self.dynamixel_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
                
                if dxl_comm_result != 0 or dxl_error != 0:
                    if self.verbose:
                        print(f"[DynamixelXH540Controller] Warning: Failed to enable torque")
                
                # Home gripper
                ADDR_GOAL_POSITION = 116  # XH540 address for goal position
                home_pos = self.max_position if self.home_to_open else self.min_position
                packet_handler.write4ByteTxRx(
                    port_handler, self.dynamixel_id, ADDR_GOAL_POSITION, home_pos)
                time.sleep(1.0)  # Wait for homing
                
                dynamixel_interface = (port_handler, packet_handler)
            else:
                # Placeholder mode - user needs to implement actual driver
                dynamixel_interface = None
                if self.verbose:
                    print("[DynamixelXH540Controller] Running in placeholder mode")
                
            # Get initial position
            if USE_DYNAMIXEL_SDK:
                ADDR_PRESENT_POSITION = 132  # XH540 address for present position
                dxl_present_position, dxl_comm_result, dxl_error = packet_handler.read4ByteTxRx(
                    port_handler, self.dynamixel_id, ADDR_PRESENT_POSITION)
                if dxl_comm_result == 0 and dxl_error == 0:
                    curr_pos = dxl_present_position
                else:
                    curr_pos = home_pos
            else:
                curr_pos = home_pos
            
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[[curr_pos,0,0,0,0,0]]
            )
            
            keep_running = True
            t_start = time.monotonic()
            iter_idx = 0
            
            while keep_running:
                # command gripper
                t_now = time.monotonic()
                dt = 1 / self.frequency
                t_target = t_now
                target_pos = pose_interp(t_target)[0]
                target_vel = (target_pos - pose_interp(t_target - dt)[0]) / dt
                
                # Clamp position to valid range
                target_pos = max(self.min_position, min(self.max_position, target_pos))
                
                # Write position to Dynamixel
                if USE_DYNAMIXEL_SDK:
                    port_handler, packet_handler = dynamixel_interface
                    ADDR_GOAL_POSITION = 116
                    ADDR_PROFILE_VELOCITY = 112
                    # Convert velocity to Dynamixel units (if needed)
                    vel_units = int(abs(target_vel) * 100)  # Scale as needed
                    vel_units = min(vel_units, 32767)  # Max velocity
                    
                    packet_handler.write4ByteTxRx(
                        port_handler, self.dynamixel_id, ADDR_GOAL_POSITION, int(target_pos))
                    packet_handler.write4ByteTxRx(
                        port_handler, self.dynamixel_id, ADDR_PROFILE_VELOCITY, vel_units)
                
                # Read state from gripper
                if USE_DYNAMIXEL_SDK:
                    port_handler, packet_handler = dynamixel_interface
                    ADDR_PRESENT_POSITION = 132
                    ADDR_PRESENT_VELOCITY = 128
                    ADDR_PRESENT_CURRENT = 126
                    
                    dxl_present_position, dxl_comm_result, dxl_error = packet_handler.read4ByteTxRx(
                        port_handler, self.dynamixel_id, ADDR_PRESENT_POSITION)
                    dxl_present_velocity, _, _ = packet_handler.read4ByteTxRx(
                        port_handler, self.dynamixel_id, ADDR_PRESENT_VELOCITY)
                    dxl_present_current, _, _ = packet_handler.read2ByteTxRx(
                        port_handler, self.dynamixel_id, ADDR_PRESENT_CURRENT)
                    
                    if dxl_comm_result == 0 and dxl_error == 0:
                        position = dxl_present_position / self.scale
                        velocity = dxl_present_velocity / self.scale
                        force = dxl_present_current  # Current as proxy for force
                    else:
                        position = target_pos / self.scale
                        velocity = 0.0
                        force = 0.0
                else:
                    # Placeholder values
                    position = target_pos / self.scale
                    velocity = target_vel / self.scale
                    force = 0.0
                
                state = {
                    'gripper_state': 0,  # 0 = moving, 1 = reached, etc.
                    'gripper_position': position,
                    'gripper_velocity': velocity,
                    'gripper_force': force,
                    'gripper_measure_timestamp': time.time(),
                    'gripper_receive_timestamp': time.time(),
                    'gripper_timestamp': time.time() - self.receive_latency
                }
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0
                
                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']
                    
                    if cmd == Command.SHUTDOWN.value:
                        keep_running = False
                        break
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pos = command['target_pos'] * self.scale
                        target_time = command['target_time']
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=[target_pos, 0, 0, 0, 0, 0],
                            time=target_time,
                            max_pos_speed=self.move_max_speed,
                            max_rot_speed=self.move_max_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    elif cmd == Command.RESTART_PUT.value:
                        t_start = command['target_time'] - time.time() + time.monotonic()
                        iter_idx = 1
                    else:
                        keep_running = False
                        break
                    
                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1
                
                # regulate frequency
                dt = 1 / self.frequency
                t_end = t_start + dt * iter_idx
                precise_wait(t_end=t_end, time_func=time.monotonic)
            
            # Disable torque on shutdown
            if USE_DYNAMIXEL_SDK:
                port_handler, packet_handler = dynamixel_interface
                ADDR_TORQUE_ENABLE = 64
                TORQUE_DISABLE = 0
                packet_handler.write1ByteTxRx(
                    port_handler, self.dynamixel_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
                port_handler.closePort()
                
        except Exception as e:
            if self.verbose:
                print(f"[DynamixelXH540Controller] Error in run loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.ready_event.set()
            if self.verbose:
                print(f"[DynamixelXH540Controller] Disconnected from gripper: {self.port}")
