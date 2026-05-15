#!/usr/bin/env python3
from cubs2_msgs.msg import AircraftControl
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, TwistStamped
from std_msgs import msg
from std_msgs.msg import String
from pathlib import Path
import yaml
import numpy as np
from types import SimpleNamespace
import casadi as ca
from cyecca.lie import SO3Quat, SO3EulerB321


def _wrap_pi(a):
    return np.arctan2(np.sin(a), np.cos(a))


class AutoControlNode(Node):
    def __init__(self) -> None:
        super().__init__("auto_control")

        # Publishers
        self.pub_control = self.create_publisher(AircraftControl, "control_auto", 10)

        # Current state
        self.aileron = 0.0
        self.elevator = 0.0
        self.throttle = 0.0
        self.rudder = 0.0
        self.mode = 0  # 0 = manual, 1 = stabilized

        # Trim values (applied as offsets to stick inputs)
        self.trim_aileron = 0.0
        self.trim_elevator = 0.0
        self.trim_throttle = 0.0
        self.trim_rudder = 0.0

        # Reference trajectory subscriber
        self.ref_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            "reference_pose",
            self.reference_pose_callback,
            10,
        )

        # True state subscriber (for feedback control)
        self.actual_pose_sub = self.create_subscription(
            PoseStamped, "/sportcub/pose", self.actual_pose_callback, 10
        )

        self.velocity_sub = self.create_subscription(
            TwistStamped, "/sportcub/velocity", self.speed_callback, 10
        )

        self.actual_data = {
            "x_est": 0.0,
            "y_est": 0.0,
            "z_est": 0.0,
            "roll_est": 0.0,
            "pitch_est": 0.0,
            "yaw_est": 0.0,
            "vx_est": 0.0,
            "vy_est": 0.0,
            "vz_est": 0.0,
            "v_est": 0.0,
            "gamma_est": 0.0,
            "vdot_est": 0.0,
            "p_est": 0.0,
            "q_est": 0.0,
            "r_est": 0.0,
        }

        self.ref_data = {
            "des_v": 0.0,
            "des_gamma": 0.0,
            "des_heading": 0.0,
            "des_a": 0.0,
            "des_phi": 0.0,
            "des_x": 0.0,
            "des_y": 0.0,
        }

        # Store reference pose
        self.ref_pose = None

        # Flight mode and timing
        self.prev_speed = 0
        self.flight_mode = "takeoff"
        self.dt = 0.01
        self.g = 9.81
        self.thr_max = 7.5  # Maximum Thrust

        self.args = "sim"  # Vehicle selection
        this_file = Path(__file__).resolve()
        self.base_dir = this_file.parent / "param"

        # TECS controller state
        self.error_norm_Es_dot_integral = 0
        self.error_dist_term_integral = 0
        self.error_pitch_integral = 0
        self.error_r_integral = 0
        self.error_r_last = 0
        self.error_xtrack_integral = 0

        # Roll controller (options: "stabilized" | "phi_stick" | "direct")
        self.roll_mode = "stabilized"
        self._phi_cmd = 0.0
        self._e_phi_int = 0.0

        # Takeoff yaw rate controller (maintain zero yaw)
        self._e_r_int = 0.0
        self._K_r_p = 0.5
        self._K_r_i = 0.1
        self._r_int_max = 0.3

        self.timer = self.create_timer(self.dt, self.control_callback)

        self.time = 0
        self.takeoff_time = 0

        self.reload_gains()

    def reload_gains(self):
        self.get_logger().info(f"[TECSControl] Loading gains from: {self.args}.yaml")
        gain_path = self.base_dir / f"{self.args}.yaml"

        if not gain_path.exists():
            raise FileNotFoundError(f"[TECSControl] Gain file not found: {gain_path}")

        with open(gain_path, "r") as f:
            raw = yaml.safe_load(f)

        self.param = SimpleNamespace(**raw)

        self.phi_lim = np.deg2rad(self.param.phi_lim_deg)
        self.chi_deadband = np.deg2rad(self.param.chi_deadband_deg)
        self.phi_dot_lim = np.deg2rad(self.param.phi_dot_lim_deg_s)

        self.mass = self.param.mass
        self.weight = self.mass * self.g

        # Load trim values from parameter file
        self.trim_aileron = getattr(self.param, "trim_aileron", 0.0)
        self.trim_elevator = getattr(self.param, "trim_elevator", 0.0)
        self.trim_throttle = getattr(self.param, "trim_throttle", 0.0)
        self.trim_rudder = getattr(self.param, "trim_rudder", 0.0)

    def _l1_guidance(self, x, y, vx, vy, V, x_ref, y_ref, psi_ref, phi_ff):
        """L1 nonlinear guidance law (Park et al. 2004).

        Places the L1 point L1_dist ahead of the reference along its heading.
        Lateral acceleration a_lat drives a bank angle command
        that corrects both cross-track error and heading error simultaneously.

        Returns phi_des, eta, a_lat, e_ct.
        """
        L1 = self.param.L1_dist

        x_L1 = x_ref + L1 * np.cos(psi_ref)
        y_L1 = y_ref + L1 * np.sin(psi_ref)

        dx = x_L1 - x
        dy = y_L1 - y

        bearing = np.arctan2(dy, dx)
        chi = np.arctan2(vy, vx)

        # ENU frame: positive roll decreases chi, so eta is negated vs. NED derivation.
        eta = _wrap_pi(chi - bearing)

        V_safe = max(V, 0.5)
        a_lat = 2.0 * V_safe**2 / L1 * np.sin(eta)

        phi_L1 = np.arctan2(a_lat, self.g)
        phi_des = self.param.K_phi_fb * phi_L1 + self.param.K_phi_ff * phi_ff

        # Signed cross-track error (+ = left of path)
        e_ct = -np.sin(psi_ref) * (x - x_ref) + np.cos(psi_ref) * (y - y_ref)

        return phi_des, eta, a_lat, e_ct

    def compute_thrust_pitch(
        self, x, y, z, ref_data, vx_est, vy_est, vz_est, V_est, gamma_est, vdot_est
    ):
        # ref data in function of time
        ref_airspeed = ref_data["des_v"]
        ref_gamma = ref_data["des_gamma"]  # Glide slope angle
        # ref_xtrack_err = ref_data['xtrack_err']
        ref_accel = ref_data["des_a"]

        r_V = float(ref_airspeed)  # desired body-frame speed
        r_gamma = float(ref_gamma)  # desired flight path angle
        r_V_dot = float(ref_accel)  # desired acceleration

        # Envelope protection: clip desired acceleration
        drag = 1.0
        r_V_dot = np.clip(
            r_V_dot, -drag / self.weight, (self.thr_max - drag) / self.weight
        )

        # -------------------Desired Thrust-------------------#
        # Thrust controls total specific energy (sum channel).
        error_norm_Es_dot = (r_gamma - gamma_est) + (r_V_dot - vdot_est) / self.g
        thrust_unsat = self.param.trim_thrust + self.weight * (
            self.param.K_thrustp * (gamma_est + vdot_est / self.g)
            + self.param.K_thrusti * self.error_norm_Es_dot_integral
        )

        thrust = float(np.clip(thrust_unsat, 0.0, self.thr_max))

        # Thrust anti-windup
        allow_I = True
        if thrust >= self.thr_max - 1e-9 and error_norm_Es_dot > 0.0:
            allow_I = False
        if thrust <= 0.0 + 1e-9 and error_norm_Es_dot < 0.0:
            allow_I = False

        if allow_I:
            self.error_norm_Es_dot_integral += error_norm_Es_dot * self.dt
            self.error_norm_Es_dot_integral = np.clip(
                self.error_norm_Es_dot_integral,
                -self.param.norm_Es_dot_integral_max,
                self.param.norm_Es_dot_integral_max,
            )

        # -------------------Desired Pitch-------------------#
        # Pitch controls energy distribution (difference channel).
        error_dist_term = (r_gamma - gamma_est) - (r_V_dot - vdot_est) / self.g
        pitch_unsat = (
            self.param.K_pitchi * self.error_dist_term_integral
            - self.param.K_pitchp * (gamma_est - vdot_est / self.g)
        )

        pitch = float(np.clip(pitch_unsat, np.deg2rad(-20), np.deg2rad(20)))

        allow_I = True
        if pitch >= np.deg2rad(20) - 1e-9 and error_dist_term > 0.0:
            allow_I = False
        if pitch <= np.deg2rad(-20) + 1e-9 and error_dist_term < 0.0:
            allow_I = False

        if allow_I:
            self.error_dist_term_integral += error_dist_term * self.dt
            self.error_dist_term_integral = np.clip(
                self.error_dist_term_integral,
                -self.param.dist_term_integral_max,
                self.param.dist_term_integral_max,
            )
        # self.get_logger().debug(f"r_gamma: {r_gamma:5.2f}, gamma_est: {gamma_est:5.2f}, r_V_dot: {r_V_dot:5.2f}, vdot_est: {vdot_est:5.2f}")

        return thrust, pitch

    def compute_control(self, ref_data, actual_data, ref_thrust=None, ref_pitch=None):
        # actual data
        x = actual_data["x_est"]
        y = actual_data["y_est"]
        z = actual_data["z_est"]
        roll = actual_data["roll_est"]
        pitch = actual_data["pitch_est"]
        yaw = actual_data["yaw_est"]
        vx_est = actual_data["vx_est"]
        vy_est = actual_data["vy_est"]
        vz_est = actual_data["vz_est"]
        V_est = actual_data["v_est"]
        gamma_est = actual_data["gamma_est"]
        vdot_est = actual_data["vdot_est"]
        p_est = actual_data["p_est"]
        q_est = actual_data["q_est"]
        r_est = actual_data["r_est"]

        # -------------------Compute Reference Outer Loop and Heading------------------#
        # Get desired thrust and pitch (we can remove this if we want to fully separate the two functions during implementation)
        if ref_thrust == None or ref_pitch == None:
            ref_thrust, ref_pitch = self.compute_thrust_pitch(
                x, y, z, ref_data, vx_est, vy_est, vz_est, V_est, gamma_est, vdot_est
            )  # Outer loop TECS controller

        r_heading = ref_data["des_heading"]

        # Elevator control: compute errors
        pitch = -1 * pitch
        error_pitch = _wrap_pi(ref_pitch - pitch)

        q_turn = np.sin(roll) * np.cos(pitch) * np.tan(roll) * self.g / V_est
        error_q = q_turn - q_est  # turning pitch
        error_q = (error_q + np.pi) % (2 * np.pi) - np.pi

        nz_excess = (1.0 / np.cos(roll)) - 1.0  # Steady-turn feed-forward
        ele_ff_phi = self.param.K_phi_elev * nz_excess

        # Integral of pitch error
        self.error_pitch_integral += error_pitch * self.dt
        if self.error_pitch_integral > self.param.pitch_integral_max:
            self.error_pitch_integral = self.param.pitch_integral_max
        elif self.error_pitch_integral < -self.param.pitch_integral_max:
            self.error_pitch_integral = -self.param.pitch_integral_max

        # Control commands for elevator
        elev_cmd = (
            self.param.trim_elevator
            + (
                self.param.K_elevp * error_pitch
                + self.param.K_elevi * self.error_pitch_integral
            )
            + self.param.K_q * error_q
        )
        elev_cmd += ele_ff_phi  # feed-forward elevator wrt to roll angle
        elev_cmd = np.clip(elev_cmd, -1, 1)  # Saturation

        # Throttle control: normalize thrust command
        thr_cmd = np.clip(ref_thrust / self.thr_max, 0.0, 1.0)

        # --- L1 Guidance (lateral outer loop) ---
        phi_des, eta, a_lat, e_ct = self._l1_guidance(
            x, y, vx_est, vy_est, V_est,
            ref_data["des_x"], ref_data["des_y"],
            r_heading, ref_data["des_phi"],
        )
        # self.get_logger().debug(f"L1 eta: {np.rad2deg(eta):5.1f} deg  e_ct: {e_ct:5.2f} m  a_lat: {a_lat:5.2f} m/s²")

        phi_des = float(np.clip(phi_des, -self.phi_lim, self.phi_lim))
        dphi_max = self.phi_dot_lim * self.dt
        phi_des = np.clip(phi_des - self._phi_cmd, -dphi_max, dphi_max) + self._phi_cmd
        self._phi_cmd = float(np.clip(phi_des, -self.phi_lim, self.phi_lim))

        # Inner loop roll control
        if self.roll_mode == "stabilized":
            # Roll stabilizer: PD on (phi, p) -> aileron
            e_phi = _wrap_pi(self._phi_cmd - roll)
            # Integrator with clamp
            self._e_phi_int += e_phi * self.dt
            self._e_phi_int = float(
                np.clip(self._e_phi_int, -self.param.i_phi_max, self.param.i_phi_max)
            )

            # Damping on measured roll-rate
            d_term = -self.param.K_phi_d * p_est

            ail_cmd = (
                self.param.trim_aileron
                + self.param.K_phi_p * e_phi
                + self.param.K_phi_i * self._e_phi_int
                + d_term
            )

            ail_cmd = float(np.clip(ail_cmd, -self.param.da_max, self.param.da_max))

        elif self.roll_mode == "phi_stick":
            # Direct bank angle command (for onboard gyro)
            ail_cmd = float(np.clip(phi_des / self.phi_lim, -1.0, 1.0))

        else:  # "direct" mode: yaw error -> aileron
            err_yaw = r_heading - yaw
            err_yaw = (err_yaw + np.pi) % (2 * np.pi) - np.pi
            error_r_deriv = (err_yaw - self.error_r_last) / self.dt
            self.error_r_last = err_yaw
            self.error_r_integral += err_yaw * self.dt
            if self.error_r_integral > self.param.r_integral_max:
                self.error_r_integral = self.param.r_integral_max
            elif self.error_r_integral < -self.param.r_integral_max:
                self.error_r_integral = -self.param.r_integral_max

            ail_cmd = (
                self.param.trim_ail
                + self.param.K_deltap * err_yaw
                + self.param.K_deltai * self.error_r_integral
                + self.param.K_deltad * error_r_deriv
            )
            ail_cmd = float(np.clip(ail_cmd, -1.0, 1.0))

        # Coordinated turn: r_des = g*tan(phi)/V drives beta to zero
        r_coord = self.g * np.tan(np.clip(roll, -np.deg2rad(60), np.deg2rad(60))) / max(V_est, 1.0)
        rud_cmd = float(np.clip(self.param.K_rud_coord * (r_coord - r_est), -1.0, 1.0))

        # Set control outputs
        self.aileron = ail_cmd
        self.elevator = elev_cmd
        self.throttle = thr_cmd
        self.rudder = rud_cmd

    def reference_pose_callback(self, msg: PoseWithCovarianceStamped):
        """Callback for reference trajectory pose."""
        self.ref_pose = msg

    def control_callback(self):
        """Publish current control state as AircraftControl message."""

        ################################### FLIGHT MODE ####################################
        flight_mode_msg = String()
        if self.actual_data["z_est"] <= 0.5:
            new_mode = "takeoff"
        else:
            new_mode = "airborne"

        if new_mode != self.flight_mode:
            self.get_logger().info(
                "Flight mode changed from: %s to %s" % (self.flight_mode, new_mode)
            )
            self.flight_mode = new_mode
            flight_mode_msg.data = new_mode

            # Reset controllers when entering takeoff mode
            if new_mode == "takeoff":
                self._e_r_int = 0.0  # Reset yaw rate integrator

        if flight_mode_msg.data == "":
            flight_mode_msg.data = self.flight_mode
        ####################################################################################

        self.time += self.dt

        if self.flight_mode == "takeoff":
            self.takeoff_time += self.dt

            # Throttle ramp with floor/ceiling
            self.throttle = ca.fmin(1.00, ca.fmax(0.7, self.throttle + 2.0 * self.dt)) # Cancel out trim in command

            self.aileron = 0.0  # Wings-level during takeoff

            # Yaw rate control: maintain zero yaw rate using rudder
            r_est = self.actual_data.get("r_est", 0.0)
            e_r = 0.0 - r_est

            self._e_r_int += e_r * self.dt
            self._e_r_int = float(
                np.clip(self._e_r_int, -self._r_int_max, self._r_int_max)
            )

            rud_cmd = (
                self._K_r_p * e_r
                + self._K_r_i * self._e_r_int
            )
            self.rudder = float(np.clip(rud_cmd, -1.0, 1.0))

            # Elevator schedule: pitch up as airspeed increases
            v_to = 0.5
            e_down = -0.02
            e_up = 0.15
            e_rate = 0.40
            if self.actual_data["v_est"] == None:
                self.actual_data["v_est"] = (
                    0.0  # Initialize V_est, assume start at stationary
                )

            self.elevator = ca.if_else(
                self.actual_data["v_est"] < v_to,
                e_down,
                ca.fmin(e_up, self.elevator + e_rate * self.dt),
            )

        if self.flight_mode == "airborne":
            planner_v = 6.0
            des_a = self.param.K_V * (planner_v - np.abs(self.actual_data["v_est"]))

            pose_q = np.array([
                self.ref_pose.pose.pose.orientation.w,
                self.ref_pose.pose.pose.orientation.x,
                self.ref_pose.pose.pose.orientation.y,
                self.ref_pose.pose.pose.orientation.z,
            ])

            SO3_pose = SO3Quat.elem(ca.horzcat(pose_q))
            SO3_321 = SO3EulerB321.from_Quat(SO3_pose).param

            des_heading  = float(SO3_321[0])
            des_pitch_ref = float(SO3_321[1])
            des_phi      = float(SO3_321[2])

            # Vertical cross-track error: correct z_desired for along-track offset so
            # the altitude error is path-perpendicular regardless of whether the aircraft
            # is ahead of or behind the reference point.
            z_desired = float(self.ref_pose.pose.pose.position.z)
            z_cur     = self.actual_data["z_est"]
            x_cur     = self.actual_data["x_est"]
            y_cur     = self.actual_data["y_est"]

            s_along  = ((x_cur - float(self.ref_pose.pose.pose.position.x)) * np.cos(des_heading)
                      + (y_cur - float(self.ref_pose.pose.pose.position.y)) * np.sin(des_heading))
            z_err    = z_cur - (z_desired + s_along * np.tan(des_pitch_ref))

            gamma_fb = np.arctan(-z_err / max(planner_v, 0.5))

            # Longitudinal feedforward from reference pitch (gamma_ref approx theta_ref for small AoA)
            gamma_ff = des_pitch_ref
            des_gamma = np.clip(
                self.param.K_gamma_fb * gamma_fb + self.param.K_gamma_ff * gamma_ff,
                -np.pi / 4, np.pi / 4,
            )

            self.ref_data = {
                "des_v": planner_v,
                "des_gamma": des_gamma,
                "des_heading": des_heading,
                "des_a": des_a,
                "des_phi": des_phi,
                "des_x": float(self.ref_pose.pose.pose.position.x),
                "des_y": float(self.ref_pose.pose.pose.position.y),
            }

            self.compute_control(self.ref_data, self.actual_data)

        msg = AircraftControl()

        msg.header.stamp = self.get_clock().now().to_msg()
        # Only apply trim in airborne mode
        if self.flight_mode == "takeoff":
            msg.aileron = float(self.aileron)
            msg.elevator = float(self.elevator)
            msg.throttle = float(self.throttle)
            msg.rudder = float(self.rudder)
        else:
            msg.aileron = float(self.aileron) + self.trim_aileron
            msg.elevator = float(self.elevator) + self.trim_elevator
            msg.throttle = float(self.throttle) + self.trim_throttle
            msg.rudder = float(self.rudder) + self.trim_rudder
        msg.mode = int(self.mode)
        self.pub_control.publish(msg)

    def speed_callback(self, msg: TwistStamped):
        """Update velocity estimates from twist message."""
        msg = msg.twist
        self.actual_data["vx_est"] = msg.linear.x
        self.actual_data["vy_est"] = msg.linear.y
        self.actual_data["vz_est"] = msg.linear.z
        v = np.linalg.norm([msg.linear.x, msg.linear.y, msg.linear.z])
        self.actual_data["v_est"] = v

        # Estimate flight path angle
        # gamma_new = np.arctan(np.clip(msg.linear.z / msg.linear.x, -1.0, 1.0))
        gamma_new = np.arctan2(msg.linear.z, np.sqrt(msg.linear.x**2 + msg.linear.y**2)) # TODO : Double check if this is correct by convention

        # self.get_logger().debug(f"Gamma: {gamma_new:.3f}")
        self.actual_data["gamma_est"] = gamma_new

        # Angular rates
        self.actual_data["p_est"] = msg.angular.x
        self.actual_data["q_est"] = msg.angular.y
        self.actual_data["r_est"] = msg.angular.z

        # Low-pass filter for acceleration estimate
        fc = 100.0
        alpha = 1.0 - np.exp(-2 * np.pi * fc * self.dt)
        vdot_raw = (v - self.prev_speed) / max(self.dt, 1e-6)
        vdot_est = self._lpf("vdot", vdot_raw, alpha)
        self.actual_data["vdot_est"] = float(vdot_est)
        self.prev_speed = v

    def actual_pose_callback(self, msg: PoseStamped):
        """Update position and attitude estimates from pose message."""
        self.actual_data["x_est"] = msg.pose.position.x
        self.actual_data["y_est"] = msg.pose.position.y
        self.actual_data["z_est"] = msg.pose.position.z

        # TODO: check
        pose_q = np.array([
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
        ])

        SO3_pose = SO3Quat.elem(ca.horzcat(pose_q))
        SO3_321 = SO3EulerB321.from_Quat(SO3_pose).param

        self.actual_data["roll_est"] = float(SO3_321[2])
        self.actual_data["pitch_est"] = float(SO3_321[1])
        self.actual_data["yaw_est"] = float(SO3_321[0])

    def _lpf(self, name: str, new_value, alpha: float):
        """
        Exponential low-pass update for <name>.
        Uses/creates attributes: <name>_est and <name>_est_last.
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")
        last_name = f"{name}_est_last"
        est_name = f"{name}_est"

        last = getattr(self, last_name, None)
        if last is None:
            last = new_value

        est = alpha * new_value + (1.0 - alpha) * last
        setattr(self, est_name, est)
        setattr(self, last_name, est)
        return est

    def _lpf_many(self, mapping: dict, alpha: float):
        """Batch low-pass filter updates."""
        for k, v in mapping.items():
            self._lpf(k, v, alpha)


def main(args=None):
    rclpy.init(args=args)
    node = AutoControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down auto control node")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
