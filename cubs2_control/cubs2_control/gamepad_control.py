#!/usr/bin/env python3
# Copyright 2025 CogniPilot Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Gamepad control node for SportCub simulation.

Maps gamepad inputs to AircraftControl message format.

Publishes:
  /control (AircraftControl): aircraft control commands

Requires:
  - ROS 2 joy package: sudo apt install ros-${ROS_DISTRO}-joy
  - Gamepad connected via USB or Bluetooth
  - joy_node running (automatically launched with this node)

Usage:
  ros2 launch fixed_wing_purt gamepad_control.xml

Axis indices (standard gamepad layout):
  0: Left stick X (left/right)
  1: Left stick Y (up/down)
  2: Right stick X (left/right)
  3: Right stick Y (up/down)
  Note: Axis values range from -1.0 to 1.0
"""
import math

from cubs2_msgs.msg import AircraftControl
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Empty


class GamepadControlNode(Node):
    def __init__(self):
        super().__init__('gamepad_control')

        # Declare button mapping parameters
        self.declare_parameter('button_reset_neutral', 0)
        self.declare_parameter('button_send_reset', 1)
        self.declare_parameter('button_trim_rudder_left', 2)
        self.declare_parameter('button_trim_rudder_right', 3)
        self.declare_parameter('button_left_bumper', 4)
        self.declare_parameter('button_right_bumper', 5)
        self.declare_parameter('button_minus', 6)
        self.declare_parameter('button_pause_toggle', 7)
        self.declare_parameter('button_exit', 8)
        self.declare_parameter('button_dpad_up', -1)
        self.declare_parameter('button_dpad_down', -1)
        self.declare_parameter('button_dpad_left', -1)
        self.declare_parameter('button_dpad_right', -1)

        # Declare axis mapping parameters
        self.declare_parameter('axis_aileron', 3)
        self.declare_parameter('axis_elevator', 4)
        self.declare_parameter('axis_rudder', 0)
        self.declare_parameter('axis_throttle', 1)
        self.declare_parameter('axis_dpad_horizontal', 6)
        self.declare_parameter('axis_dpad_vertical', 7)
        self.declare_parameter('axis_left_trigger', 2)
        self.declare_parameter('axis_right_trigger', 5)

        # Declare inversion parameters
        self.declare_parameter('invert_aileron', False)
        self.declare_parameter('invert_elevator', False)
        self.declare_parameter('invert_rudder', False)
        self.declare_parameter('invert_throttle', False)

        # Declare control parameters
        self.declare_parameter('throttle_default', 0.0)
        self.declare_parameter('deadzone', 0.05)
        self.declare_parameter('trim_step', 0.01)
        self.declare_parameter('throttle_step', 0.02)
        self.declare_parameter('throttle_deadzone', 0.2)
        self.declare_parameter('throttle_exponent', 3.0)
        self.declare_parameter('aileron_exponent', 1.0)
        self.declare_parameter('elevator_exponent', 1.0)

        # D-pad configuration
        self.declare_parameter('dpad_is_buttons', False)
        self.declare_parameter('dpad_deadzone', 0.5)

        # Get button mappings
        self.btn_reset_neutral = self.get_parameter(
            'button_reset_neutral').value
        self.btn_send_reset = self.get_parameter('button_send_reset').value
        self.btn_trim_rud_left = self.get_parameter(
            'button_trim_rudder_left').value
        self.btn_trim_rud_right = self.get_parameter(
            'button_trim_rudder_right').value
        self.btn_left_bumper = self.get_parameter('button_left_bumper').value
        self.btn_right_bumper = self.get_parameter('button_right_bumper').value
        self.btn_minus = self.get_parameter('button_minus').value
        self.btn_pause_toggle = self.get_parameter('button_pause_toggle').value
        self.btn_exit = self.get_parameter('button_exit').value
        self.btn_dpad_up = self.get_parameter('button_dpad_up').value
        self.btn_dpad_down = self.get_parameter('button_dpad_down').value
        self.btn_dpad_left = self.get_parameter('button_dpad_left').value
        self.btn_dpad_right = self.get_parameter('button_dpad_right').value

        # Get axis mappings
        self.axis_throttle = self.get_parameter('axis_throttle').value
        self.axis_rudder = self.get_parameter('axis_rudder').value
        self.axis_elevator = self.get_parameter('axis_elevator').value
        self.axis_aileron = self.get_parameter('axis_aileron').value
        self.axis_dpad_h = self.get_parameter('axis_dpad_horizontal').value
        self.axis_dpad_v = self.get_parameter('axis_dpad_vertical').value
        self.axis_left_trigger = self.get_parameter('axis_left_trigger').value
        self.axis_right_trigger = self.get_parameter(
            'axis_right_trigger').value

        # Get inversion flags
        self.invert_elevator = self.get_parameter('invert_elevator').value
        self.invert_aileron = self.get_parameter('invert_aileron').value
        self.invert_rudder = self.get_parameter('invert_rudder').value
        self.invert_throttle = self.get_parameter('invert_throttle').value

        # Get control parameters
        self.throttle_default = self.get_parameter('throttle_default').value
        self.deadzone = self.get_parameter('deadzone').value
        self.trim_step = self.get_parameter('trim_step').value
        self.throttle_step = self.get_parameter('throttle_step').value
        self.throttle_deadzone = self.get_parameter('throttle_deadzone').value
        self.throttle_exponent = self.get_parameter('throttle_exponent').value
        self.aileron_exponent = self.get_parameter('aileron_exponent').value
        self.elevator_exponent = self.get_parameter('elevator_exponent').value
        self.dpad_is_buttons = self.get_parameter('dpad_is_buttons').value
        self.dpad_deadzone = self.get_parameter('dpad_deadzone').value

        # Publishers
        self.pub_control = self.create_publisher(
            AircraftControl, '/control', 10)
        self.pub_reset = self.create_publisher(Empty, '/reset', 10)
        self.pub_pause = self.create_publisher(Empty, '/pause', 10)

        # Subscriber to joy messages with QoS for no queuing
        from rclpy.qos import QoSHistoryPolicy
        from rclpy.qos import QoSProfile
        from rclpy.qos import QoSReliabilityPolicy

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,  # Only keep the latest message
        )
        self.sub_joy = self.create_subscription(
            Joy, '/joy', self.joy_callback, qos_profile)

        # Subscribe to external control messages (from RViz dropdown, etc.)
        # to sync mode changes from other sources
        self.sub_control = self.create_subscription(
            AircraftControl, '/control', self.control_callback, 10)

        # Current state
        self.aileron = 0.0
        self.elevator = 0.0
        self.throttle = self.throttle_default
        self.rudder = 0.0
        self.mode = 0  # 0 = manual, 1 = stabilized

        # Trim values (applied as offsets to stick inputs)
        self.trim_aileron = 0.0
        self.trim_elevator = 0.0
        self.trim_throttle = 0.0
        self.trim_rudder = 0.0

        # Button state tracking (for edge detection)
        self.last_buttons = []

        # Trim button hold tracking (for acceleration)
        self.trim_hold_count = {}  # button_index -> count of consecutive holds

        # D-pad axis state tracking (for edge detection on axes)
        self.last_dpad_h = 0.0
        self.last_dpad_v = 0.0

        # Throttle axis state tracking (for spring-loaded increment mode)
        self.last_throttle_axis = 0.0
        self.throttle_hold_count_up = 0
        self.throttle_hold_count_down = 0

        # Throttle axis state tracking (for spring-loaded stick increment mode)
        self.last_throttle_axis = 0.0
        self.throttle_hold_count = 0

        # Time tracking for rate-independent throttle response
        self.last_joy_time = None

        self.get_logger().info('Gamepad control node started')
        self.get_logger().info('Control mode: RC Transmitter style (spring-loaded throttle)')
        self.get_logger().info('Axis mapping:')
        self.get_logger().info(
            f'  Left stick Y (up/down)    -> Throttle bump up/down (axis {
                self.axis_throttle})'
        )
        self.get_logger().info(
            f'  Left stick X (left/right) -> Rudder (axis {self.axis_rudder})')
        self.get_logger().info(
            f'  Right stick Y (up/down)   -> Elevator (axis {
                self.axis_elevator})'
        )
        self.get_logger().info(
            f'  Right stick X (left/right)-> Aileron (axis {self.axis_aileron})')
        self.get_logger().info('Button mapping:')
        self.get_logger().info(
            f'  Button {self.btn_reset_neutral} (A): Reset to neutral')
        self.get_logger().info(
            f'  Button {self.btn_send_reset} (B): Send /reset')
        self.get_logger().info(
            f'  Button {self.btn_trim_rud_left} (X): Trim rudder left')
        self.get_logger().info(
            f'  Button {self.btn_trim_rud_right} (Y): Trim rudder right')
        self.get_logger().info(
            f'  Button {self.btn_right_bumper}: Toggle flight mode (manual/stabilized)')
        dpad_type = (
            'buttons' if self.dpad_is_buttons else f'axes ({
                self.axis_dpad_h}, {
                self.axis_dpad_v})')
        self.get_logger().info(f'  D-pad ({dpad_type}): Trim elevator/aileron')
        self.get_logger().info(f'  Button {self.btn_exit}: Exit')
        self.get_logger().info(
            f'  Button {self.btn_pause_toggle}: Toggle /pause')

    def apply_deadzone(self, value: float) -> float:
        """Apply deadzone to axis value."""
        if abs(value) < self.deadzone:
            return 0.0
        return value

    def _button_pressed(self, msg: Joy, button_idx: int) -> bool:
        """Check if button was just pressed (edge detection)."""
        if button_idx < 0 or button_idx >= len(msg.buttons):
            return False
        if button_idx >= len(self.last_buttons):
            return False
        return msg.buttons[button_idx] and not self.last_buttons[button_idx]

    def _button_held(self, msg: Joy, button_idx: int) -> bool:
        """Check if button is currently held down."""
        if button_idx < 0 or button_idx >= len(msg.buttons):
            return False
        return msg.buttons[button_idx]

    def _get_trim_step(self, button_idx: int) -> float:
        """
        Get trim step size with acceleration for held buttons.

        First press: normal step
        After 5 presses: 2x speed
        After 15 presses: 4x speed
        After 30 presses: 8x speed

        """
        # Increment hold count
        if button_idx not in self.trim_hold_count:
            self.trim_hold_count[button_idx] = 0
        self.trim_hold_count[button_idx] += 1

        count = self.trim_hold_count[button_idx]

        # Accelerate based on hold duration
        if count > 30:
            return self.trim_step * 8.0
        elif count > 15:
            return self.trim_step * 4.0
        elif count > 5:
            return self.trim_step * 2.0
        else:
            return self.trim_step

    def _get_trim_step_axis(self, axis_key: str, is_active: bool) -> float:
        """
        Get trim step size with acceleration for held axis positions.

        Similar to button acceleration but uses string keys for axis directions.

        """
        if not is_active:
            return self.trim_step

        # Increment hold count
        if axis_key not in self.trim_hold_count:
            self.trim_hold_count[axis_key] = 0
        self.trim_hold_count[axis_key] += 1

        count = self.trim_hold_count[axis_key]

        # Accelerate based on hold duration
        if count > 30:
            return self.trim_step * 8.0
        elif count > 15:
            return self.trim_step * 4.0
        elif count > 5:
            return self.trim_step * 2.0
        else:
            return self.trim_step

    def _get_throttle_step(self, direction: str) -> float:
        """
        Get throttle step size with acceleration for held stick.

        Parameters
        ----------
        direction : str
            Direction of throttle change ('up' or 'down')

        Slower acceleration than trim controls for better fine control.

        """
        # Use separate counters for up and down
        if direction == 'up':
            self.throttle_hold_count_up += 1
            count = self.throttle_hold_count_up
        else:
            self.throttle_hold_count_down += 1
            count = self.throttle_hold_count_down

        # Slower acceleration than trim: longer delays before speed-up
        if count > 60:  # ~3 seconds at 20Hz
            return self.throttle_step * 4.0
        elif count > 30:  # ~1.5 seconds
            return self.throttle_step * 2.0
        elif count > 10:  # ~0.5 seconds
            return self.throttle_step * 1.5
        else:
            return self.throttle_step

    def joy_callback(self, msg: Joy):
        """Process joystick messages."""
        # Calculate time delta for rate-independent response
        current_time = self.get_clock().now()
        if self.last_joy_time is None:
            dt = 0.02  # Assume 50Hz for first message
        else:
            dt = (current_time - self.last_joy_time).nanoseconds / 1e9
            # Clamp to reasonable range (10-1000 Hz)
            dt = max(0.001, min(dt, 0.1))
        self.last_joy_time = current_time

        # Extract axis values with safety checks
        def get_axis(index: int) -> float:
            if index < len(msg.axes):
                return self.apply_deadzone(float(msg.axes[index]))
            return 0.0

        # Map axes to controls
        # Left stick Y -> Throttle increment/decrement (spring-loaded stick mode)
        # Joy axes: down = -1, up = 1
        # Stick up increases throttle, stick down decreases throttle
        # Rate is proportional to stick deflection
        throttle_raw = get_axis(self.axis_throttle)
        if self.invert_throttle:
            throttle_raw = -throttle_raw

        # Increment/decrement throttle with exponential rate based on stick
        # deflection
        if abs(
                throttle_raw) > self.throttle_deadzone:  # Threshold to avoid accidental bumps
            # Exponential factor: raise deflection to configurable power
            # Higher exponent = finer control near center, much faster at
            # extremes
            deflection_factor = abs(throttle_raw) ** self.throttle_exponent

            if throttle_raw > self.throttle_deadzone:  # Stick pushed up
                step = self._get_throttle_step(
                    'up') * deflection_factor * (dt / 0.02)
                self.throttle += step
                self.throttle_hold_count_down = 0  # Reset down counter
            else:  # Stick pushed down (throttle_raw < -0.2)
                step = self._get_throttle_step(
                    'down') * deflection_factor * (dt / 0.02)
                self.throttle -= step
                self.throttle_hold_count_up = 0  # Reset up counter
        else:
            # Reset both hold counts when stick is centered
            self.throttle_hold_count_up = 0
            self.throttle_hold_count_down = 0

        # Apply trim and clamp throttle to [0, 1]
        self.throttle = max(0.0, min(1.0, self.throttle + self.trim_throttle))
        self.last_throttle_axis = throttle_raw

        # Left stick X -> Rudder [-1, 1]
        rudder_raw = get_axis(self.axis_rudder)
        if self.invert_rudder:
            rudder_raw = -rudder_raw
        self.rudder = max(-1.0, min(1.0, rudder_raw + self.trim_rudder))

        # Right stick Y -> Elevator [-1, 1]
        # Joy Y: down = -1, up = 1, elevator positive = pitch up
        # stick up (positive) = pitch up (positive)
        elevator_raw = get_axis(self.axis_elevator)
        if self.invert_elevator:
            elevator_raw = -elevator_raw
        # Apply exponential response while preserving sign
        if self.elevator_exponent != 1.0:
            elevator_sign = 1.0 if elevator_raw >= 0.0 else -1.0
            elevator_raw = elevator_sign * \
                (abs(elevator_raw) ** self.elevator_exponent)
        self.elevator = max(-1.0, min(1.0, elevator_raw + self.trim_elevator))

        # Right stick X -> Aileron [-1, 1]
        aileron_raw = get_axis(self.axis_aileron)
        if self.invert_aileron:
            aileron_raw = -aileron_raw
        # Apply exponential response while preserving sign
        if self.aileron_exponent != 1.0:
            aileron_sign = 1.0 if aileron_raw >= 0.0 else -1.0
            aileron_raw = aileron_sign * \
                (abs(aileron_raw) ** self.aileron_exponent)
        self.aileron = max(-1.0, min(1.0, aileron_raw + self.trim_aileron))

        # Handle buttons (edge detection - trigger on press, not hold)
        if len(msg.buttons) > 0:
            # Initialize last_buttons if needed
            if len(self.last_buttons) == 0:
                self.last_buttons = [0] * len(msg.buttons)

            # Button: Reset to neutral (including trim)
            if self._button_pressed(msg, self.btn_reset_neutral):
                self.aileron = 0.0
                self.elevator = 0.0
                self.throttle = self.throttle_default
                self.rudder = 0.0
                self.trim_aileron = 0.0
                self.trim_elevator = 0.0
                self.trim_throttle = 0.0
                self.trim_rudder = 0.0
                self.get_logger().info('Reset to neutral (trim cleared)')

            # Button: Send /reset
            if self._button_pressed(msg, self.btn_send_reset):
                self.pub_reset.publish(Empty())
                self.get_logger().info('Sent /reset')

            # Button: Trim rudder left
            if self._button_held(msg, self.btn_trim_rud_left):
                step = self._get_trim_step(self.btn_trim_rud_left)
                self.trim_rudder -= step
                self.get_logger().info(
                    f'Trim rudder: {
                        self.trim_rudder:.3f} rad ({
                        math.degrees(
                            self.trim_rudder):.2f}°)')
            else:
                self.trim_hold_count[self.btn_trim_rud_left] = 0

            # Button: Trim rudder right
            if self._button_held(msg, self.btn_trim_rud_right):
                step = self._get_trim_step(self.btn_trim_rud_right)
                self.trim_rudder += step
                self.get_logger().info(
                    f'Trim rudder: {
                        self.trim_rudder:.3f} rad ({
                        math.degrees(
                            self.trim_rudder):.2f}°)')
            else:
                self.trim_hold_count[self.btn_trim_rud_right] = 0

            # D-pad trim controls (button-based or axis-based)
            if self.dpad_is_buttons:
                # D-pad uses buttons
                if self._button_held(msg, self.btn_dpad_up):
                    step = self._get_trim_step(self.btn_dpad_up)
                    self.trim_elevator -= step
                    self.get_logger().info(
                        f'Trim elevator: {
                            self.trim_elevator:.3f} rad ({
                            math.degrees(
                                self.trim_elevator):.2f}°)')
                else:
                    self.trim_hold_count[self.btn_dpad_up] = 0

                if self._button_held(msg, self.btn_dpad_down):
                    step = self._get_trim_step(self.btn_dpad_down)
                    self.trim_elevator += step
                    self.get_logger().info(
                        f'Trim elevator: {
                            self.trim_elevator:.3f} rad ({
                            math.degrees(
                                self.trim_elevator):.2f}°)')
                else:
                    self.trim_hold_count[self.btn_dpad_down] = 0

                if self._button_held(msg, self.btn_dpad_left):
                    step = self._get_trim_step(self.btn_dpad_left)
                    self.trim_aileron -= step
                    self.get_logger().info(
                        f'Trim aileron: {
                            self.trim_aileron:.3f} rad ({
                            math.degrees(
                                self.trim_aileron):.2f}°)')
                else:
                    self.trim_hold_count[self.btn_dpad_left] = 0

                if self._button_held(msg, self.btn_dpad_right):
                    step = self._get_trim_step(self.btn_dpad_right)
                    self.trim_aileron += step
                    self.get_logger().info(
                        f'Trim aileron: {
                            self.trim_aileron:.3f} rad ({
                            math.degrees(
                                self.trim_aileron):.2f}°)')
                else:
                    self.trim_hold_count[self.btn_dpad_right] = 0

            # Button: Toggle flight mode (manual/stabilized)
            if self._button_pressed(msg, self.btn_right_bumper):
                self.mode = 1 - self.mode  # Toggle between 0 and 1
                mode_name = 'STABILIZED' if self.mode == 1 else 'MANUAL'
                self.get_logger().info(f'Flight mode changed to: {mode_name}')

            # Button: Exit node
            if self._button_pressed(msg, self.btn_exit):
                self.get_logger().info('Exit button pressed, shutting down...')
                raise KeyboardInterrupt()

            # Button: Toggle pause
            if self._button_pressed(msg, self.btn_pause_toggle):
                self.pub_pause.publish(Empty())
                self.get_logger().info('Sent /pause toggle')

            # Update button state
            self.last_buttons = list(msg.buttons)

        # Handle D-pad trim controls (axis-based)
        if not self.dpad_is_buttons and self.axis_dpad_h >= 0 and self.axis_dpad_v >= 0:
            dpad_h = get_axis(self.axis_dpad_h)
            dpad_v = get_axis(self.axis_dpad_v)

            # Trim elevator with vertical D-pad
            if dpad_v > self.dpad_deadzone:
                step = self._get_trim_step_axis(
                    'dpad_v_up', dpad_v > self.dpad_deadzone)
                self.trim_elevator -= step
                if abs(dpad_v - self.last_dpad_v) > 0.1:  # Only log on change
                    self.get_logger().info(
                        f'Trim elevator: {
                            self.trim_elevator:.3f} rad ({
                            math.degrees(
                                self.trim_elevator):.2f}°)')
            elif dpad_v < -self.dpad_deadzone:
                step = self._get_trim_step_axis(
                    'dpad_v_down', dpad_v < -self.dpad_deadzone)
                self.trim_elevator += step
                if abs(dpad_v - self.last_dpad_v) > 0.1:
                    self.get_logger().info(
                        f'Trim elevator: {
                            self.trim_elevator:.3f} rad ({
                            math.degrees(
                                self.trim_elevator):.2f}°)')
            else:
                self.trim_hold_count['dpad_v_up'] = 0
                self.trim_hold_count['dpad_v_down'] = 0

            # Trim aileron with horizontal D-pad
            if dpad_h > self.dpad_deadzone:
                step = self._get_trim_step_axis(
                    'dpad_h_left', dpad_h > self.dpad_deadzone)
                self.trim_aileron -= step  # Left is positive, so subtract for left trim
                if abs(dpad_h - self.last_dpad_h) > 0.1:
                    self.get_logger().info(
                        f'Trim aileron: {
                            self.trim_aileron:.3f} rad ({
                            math.degrees(
                                self.trim_aileron):.2f}°)')
            elif dpad_h < -self.dpad_deadzone:
                step = self._get_trim_step_axis(
                    'dpad_h_right', dpad_h < -self.dpad_deadzone)
                self.trim_aileron += step  # Right is negative, so add for right trim
                if abs(dpad_h - self.last_dpad_h) > 0.1:
                    self.get_logger().info(
                        f'Trim aileron: {
                            self.trim_aileron:.3f} rad ({
                            math.degrees(
                                self.trim_aileron):.2f}°)')
            else:
                self.trim_hold_count['dpad_h_left'] = 0
                self.trim_hold_count['dpad_h_right'] = 0

            self.last_dpad_h = dpad_h
            self.last_dpad_v = dpad_v

        # Publish control messages
        self.publish_controls()

    def control_callback(self, msg: AircraftControl):
        """
        Listen for external control messages (e.g., from RViz dropdown).
        
        Syncs mode changes from other sources so both gamepad and dropdown
        control the same mode.
        """
        # Update mode if it changed externally (e.g., from RViz dropdown)
        external_mode = int(msg.mode)
        if external_mode != self.mode:
            self.mode = external_mode
            mode_name = 'STABILIZED' if self.mode == 1 else 'MANUAL'
            self.get_logger().info(f'Flight mode synced to: {mode_name}')

    def publish_controls(self):
        """Publish current control state as AircraftControl message."""
        msg = AircraftControl()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.aileron = float(self.aileron)
        msg.elevator = float(self.elevator)
        msg.throttle = float(self.throttle)
        msg.rudder = float(self.rudder)
        msg.mode = int(self.mode)
        self.pub_control.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GamepadControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down gamepad control node')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
