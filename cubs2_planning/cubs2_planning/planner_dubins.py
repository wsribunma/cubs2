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
from cyecca.planning.dubins import derive_dubins
from cyecca.planning.dubins import DubinsPathType
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist

from nav_msgs.msg import Path
import numpy as np
from racecourse_description.factory import MarkerFactory
from racecourse_description.loader import RacecourseLoader
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler
from visualization_msgs.msg import MarkerArray
from dubins_offset import (
    plan_dubins_path,
    evaluate_dubins_path,
    compute_curvature_from_xy,
    extract_dubins_segments,
    apply_polynomial_offset_to_dubins,
    rolling_median_replace,
)
from polynomial_optimization import run_poly_optimization
from tf2_ros import Buffer, TransformListener, LookupException

from cyecca.lie import SE3Quat
import casadi as ca


class DubinsGatePlannerNode(Node):
    def __init__(self):
        super().__init__("dubins_gate_planner")

        self.declare_parameter(
            "racecourse_yaml", "package://racecourse_description/config/racecourse.yaml"
        )
        self.declare_parameter("turn_radius", 5.0)
        self.declare_parameter("sample_points", 200)
        self.declare_parameter("show_turn_circles", False)
        self.declare_parameter("show_gate_turn_circles", False)
        self.declare_parameter("show_headings", False)
        self.declare_parameter("show_path_marker", False)
        self.declare_parameter("planner.heading_spacing", 20)
        self.declare_parameter("planner.altitude", 2.0)
        self.declare_parameter(
            "planner.gate_sequence", [0, 1, 2, 4, 3, 1, 2, 4, 3, 1, 2, 4, 3, 0]
        )  # Default sequence
        # m/s - velocity along trajectory
        self.declare_parameter("planner.velocity", 6.0)
        self.declare_parameter("reference_frame_id", "reference")  # TF frame name
        self.declare_parameter("max_distance", 3)

        self.marker_pub = self.create_publisher(MarkerArray, "dubins_trajectory", 10)
        self.path_pub = self.create_publisher(Path, "dubins_path", 10)

        # TF broadcaster for reference trajectory position
        self.tf_broadcaster = TransformBroadcaster(self)

        yaml_path = self.get_parameter("racecourse_yaml").value
        self.racecourse = RacecourseLoader(yaml_path)

        self.get_logger().debug(f"Loaded {len(self.racecourse.gates)} gates")

        self.plan_fn, self.eval_fn = derive_dubins()
        self.segments = self.plan_trajectory()

        # Flatten all trajectory points for time-based lookup
        self.trajectory_points = []
        for seg in self.segments:
            self.trajectory_points.extend(seg["points"])

        # Calculate total arc length and time for each point
        self._compute_trajectory_timing()

        # Record start time
        self.start_time = self.get_clock().now()

        self.timer = self.create_timer(1.0, self.publish_visualization)
        self.tf_timer = self.create_timer(
            0.02, self.publish_reference_tf
        )  # 50 Hz TF updates

        self.get_logger().info("started")
        self.previous_time = self.start_time
        self.elapsed_time = 0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Change these to your frame names
        self.frame_ref = self.get_parameter("reference_frame_id").value
        self.frame_true = "vehicle"

        # Publish Pose for reference
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, "reference_pose", 10
        )
        self.relative_pose_position = np.array([0.0, 0.0, 0.0])

        # Timer — check at 30 Hz
        self.timer = self.create_timer(1.0 / 100.0, self.update_distance)
        self.dist = 0

        self.speed_pub = self.create_publisher(Twist, "reference_speed", 10)

    def update_distance(self):
        try:
            # lookup_transform(target_frame, source_frame, time)
            tf_ref = self.tf_buffer.lookup_transform(
                self.racecourse.frame_id, self.frame_ref, rclpy.time.Time()
            )
            tf_true = self.tf_buffer.lookup_transform(
                self.racecourse.frame_id, self.frame_true, rclpy.time.Time()
            )
        except LookupException:
            self.get_logger().warn("TF not available yet", throttle_duration_sec=5.0)
            return

        # Extract XYZ
        ax = tf_ref.transform.translation.x
        ay = tf_ref.transform.translation.y
        az = tf_ref.transform.translation.z

        bx = tf_true.transform.translation.x
        by = tf_true.transform.translation.y
        bz = tf_true.transform.translation.z

        ref = SE3Quat.elem(
            ca.DM(
                np.array(
                    [
                        ax,
                        ay,
                        az,
                        tf_ref.transform.rotation.w,
                        tf_ref.transform.rotation.x,
                        tf_ref.transform.rotation.y,
                        tf_ref.transform.rotation.z,
                    ]
                )
            )
        )
        true = SE3Quat.elem(
            ca.DM(
                np.array(
                    [
                        bx,
                        by,
                        bz,
                        tf_ref.transform.rotation.w,
                        tf_true.transform.rotation.x,
                        tf_true.transform.rotation.y,
                        tf_true.transform.rotation.z,
                    ]
                )
            )
        )

        # Put true state in ref frame
        true_in_ref = SE3Quat.product(ref.inverse(), true)
        true_in_ref_pos = true_in_ref.param
        true_in_ref_pos = np.array(ca.DM(true_in_ref_pos)).flatten()

        self.relative_pose_position = true_in_ref_pos[:3]

        # Euclidean distance
        self.dist = np.linalg.norm(np.array([ax - bx, ay - by, az - bz]))

        trel = TransformStamped()
        current_time = self.get_clock().now()
        trel.header.stamp = current_time.to_msg()
        trel.header.frame_id = self.get_parameter("reference_frame_id").value
        trel.child_frame_id = "error_relative_to_reference"
        trel.transform.translation.x = true_in_ref_pos[0]
        trel.transform.translation.y = true_in_ref_pos[1]
        trel.transform.translation.z = true_in_ref_pos[2]
        trel.transform.rotation.w = true_in_ref_pos[3]
        trel.transform.rotation.x = true_in_ref_pos[4]
        trel.transform.rotation.y = true_in_ref_pos[5]
        trel.transform.rotation.z = true_in_ref_pos[6]

        self.tf_broadcaster.sendTransform(trel)

    def plan_trajectory(self):
        R = self.get_parameter("turn_radius").value
        n = self.get_parameter("sample_points").value
        gate_sequence = self.get_parameter("planner.gate_sequence").value

        all_gates = self.racecourse.gates

        # Build ordered gate list from sequence
        try:
            ordered_gates = [all_gates[idx] for idx in gate_sequence]
        except IndexError as e:
            self.get_logger().error(f"Invalid gate index in sequence: {e}")
            self.get_logger().error(f"Available gates: 0-{len(all_gates) - 1}")
            return []

        segments = []

        for i in range(len(ordered_gates) - 1):
            g0, g1 = ordered_gates[i], ordered_gates[i + 1]

            p0 = np.array([g0.center[0], g0.center[1]])
            p1 = np.array([g1.center[0], g1.center[1]])

            cost, ptype, a1, d, a2, tp0, tp1, c0, c1 = self.plan_fn(
                p0, g0.yaw, p1, g1.yaw, R
            )

            c0_arr = np.array(c0).flatten()
            c1_arr = np.array(c1).flatten()

            points = []
            for s in np.linspace(0, 1, n):
                x, y, psi = self.eval_fn(
                    s,
                    p0,
                    g0.yaw,
                    a1,
                    d,
                    a2,
                    np.array(tp0).flatten(),
                    np.array(tp1).flatten(),
                    c0_arr,
                    c1_arr,
                    R,
                )
                z = g0.center[2] + (g1.center[2] - g0.center[2]) * s
                points.append({"pos": [float(x), float(y), z], "psi": float(psi)})

            segments.append(
                {
                    "points": points,
                    "centers": [c0_arr, c1_arr],
                    "z": g0.center[0],
                }
            )

            # self.get_logger().debug(
            #     f'{g0.name} → {g1.name}: type={DubinsPathType.name(ptype)}, cost={
            #         float(cost):.2f}'
            # )

        waypoints = []
        for i, gate in enumerate(ordered_gates):
            waypoints.append({"pos": gate.center, "yaw": gate.yaw, "name": f"WP{i}"})

        turn_radius = R
        num_points_per_segment = n

        # Plan Dubins paths between consecutive waypoints
        dubins_paths = []
        dubins_plans = []

        segment_L = []
        segment_direction = []
        for i in range(len(waypoints) - 1):
            wp_start = waypoints[i]
            wp_end = waypoints[i + 1]

            # Plan Dubins path
            plan = plan_dubins_path(
                wp_start["pos"],
                wp_start["yaw"],
                wp_end["pos"],
                wp_end["yaw"],
                turn_radius,
            )
            dubins_plans.append(plan)

            # Evaluate path
            path_points = evaluate_dubins_path(
                plan,
                wp_start["pos"],
                wp_start["yaw"],
                wp_end["pos"],
                wp_end["yaw"],
                turn_radius,
                num_points=num_points_per_segment,
            )
            dubins_paths.append(path_points)

            # Calculate segment lengths from evaluated path
            path_type = plan["type"]
            total_seg_length = 0
            for j in range(1, len(path_points)):
                total_seg_length += np.linalg.norm(
                    path_points[j, :2] - path_points[j - 1, :2]
                )

            # Use the plan's a1, d, a2 values (angles and straight distance)
            # Convert to scalars if they're arrays
            a1 = float(np.atleast_1d(plan["a1"])[0])
            d = float(np.atleast_1d(plan["d"])[0])
            a2 = float(np.atleast_1d(plan["a2"])[0])

            # Calculate arc lengths from angles
            L1 = a1 * turn_radius  # Arc 1 length
            L2 = d  # Straight length
            L3 = a2 * turn_radius  # Arc 2 length

            segment_L.append(L1)
            segment_L.append(L2)
            segment_L.append(L3)

            a1_deg = float(np.degrees(a1))
            a2_deg = float(np.degrees(a2))

            segment_direction.append("L" if path_type[0] == "L" else "R")
            segment_direction.append("S")
            segment_direction.append("L" if path_type[2] == "L" else "R")

            # Combine all segments into one continuous path
            full_path = np.vstack(dubins_paths)

            # Right plot: Heading angle vs arc length
            arc_lengths = np.zeros(len(full_path))
            for i in range(1, len(full_path)):
                arc_lengths[i] = arc_lengths[i - 1] + np.linalg.norm(
                    full_path[i, :2] - full_path[i - 1, :2]
                )

            arc_lengths_norm = arc_lengths  # / arc_lengths[-1]
            # Mark waypoint transitions
            segment_boundaries = []
            current_pos = 0
            # for i, path in enumerate(dubins_paths[:-1]):
            #     current_pos += len(path) #/ len(full_path)
            #     ax2.axvline(current_pos, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
            #     ax2.text(current_pos, ax2.get_ylim()[1] * 0.95, f'WP {i+1}',
            #             rotation=90, fontsize=10, ha='right', color='red')

        segments_count = (len(waypoints) - 1) * 3

        boundary_positions = [
            [*([0, None, None] * len(waypoints)), 0]
        ]  # Position: start=0, end=
        boundary_velocities = [
            [0, *([None] * (segments_count - 1)), 0]
        ]  # Velocity: start=0, others free

        if segment_direction[0] == "R":
            start_condition = 1 / R
        elif segment_direction[0] == "L":
            start_condition = -1 / R
        else:
            assert -1 > 0  # "PANNIC"

        if segment_direction[-1] == "R":
            end_condition = 1 / R
        elif segment_direction[-1] == "L":
            end_condition = -1 / R
        else:
            assert -1 > 0  # "PANNIC"

        boundary_accelerations = [
            [start_condition, *([None] * (segments_count - 1)), end_condition]
        ]  # Acceleration: start=0

        boundary_jerk = [[None] * (segments_count + 1)]  # Jerk: start=0
        boundary_snap = [[None] * (segments_count + 1)]  # Snap: start=0
        boundary_crackle = [[None] * (segments_count + 1)]  # Snap: start=0
        boundary_pop = [[None] * (segments_count + 1)]  # Snap: start=0
        boundary_lock = [[None] * (segments_count + 1)]  # Snap: start=0

        boundary_conditions = [
            boundary_positions,
            boundary_velocities,
            boundary_accelerations,
            boundary_jerk,
            boundary_snap,
            boundary_crackle,
            boundary_pop,
            boundary_lock,
        ]

        continuity_changes = []
        i = 0
        for segment in segment_direction[0:-1]:

            current_segment = segment
            next_segment = segment_direction[i + 1]

            value = None
            match current_segment:
                case "R":
                    if next_segment == "R":
                        value = 0
                    elif next_segment == "L":
                        value = -2 / R
                    elif next_segment == "S":
                        value = -1 / R
                case "L":
                    if next_segment == "R":
                        value = 2 / R
                    elif next_segment == "L":
                        value = 0
                    elif next_segment == "S":
                        value = 1 / R
                case "S":
                    if next_segment == "R":
                        value = 1 / R
                    elif next_segment == "L":
                        value = -1 / R
                    elif next_segment == "S":
                        value = 0
                case _:
                    assert -1 > 0
            if value != 0:
                continuity_changes.append([2 + i * 5, value])  # Default to 0
            i += 1

        tau = np.abs(segment_L)  # Segment durations

        # Solve optimization problem

        outputs = run_poly_optimization(
            order=7,
            tau=tau,
            segments=segments_count,
            weights=[
                0.1,
                0.1,
                0.1,
                0.1,
                100000,
                0.1,
                0.1,
                0.1,
            ],  # Minimize snap (4th derivative)
            boundary_conditions=boundary_conditions,
            continuity=continuity_changes,
        )

        offsetted_dubins, arc_lengths, lateral_offsets = (
            apply_polynomial_offset_to_dubins(full_path, outputs["polys"], tau)
        )

        pts = offsetted_dubins
        dx = np.diff(pts[:, 0])
        dy = np.diff(pts[:, 1])

        pts = offsetted_dubins

        dx = np.diff(pts[:, 0])
        dy = np.diff(pts[:, 1])
        dx = np.append(dx, dx[-1])
        dy = np.append(dy, dy[-1])

        heading = rolling_median_replace(np.arctan2(dy, dx), thresh=0.1)

        segments_new = []

        i = 0
        for segment in segments:
            new_points = []
            for point in segment["points"]:
                new_points.append(
                    {
                        "pos": [float(pts[i, 0]), float(pts[i, 1]), point["pos"][2]],
                        "psi": float(heading[i]),
                    }
                )
                i += 1
            segments_new.append(
                {
                    "points": new_points,
                    "centers": segment["centers"],
                    "z": segment["z"],
                }
            )

        return segments_new

    def _compute_trajectory_timing(self):
        """Calculate arc length, time, and heading rate for each trajectory point."""
        if not self.trajectory_points:
            return

        velocity = self.get_parameter("planner.velocity").value
        g = 9.81  # gravity (m/s^2)

        # Add arc length and time to each point
        cumulative_length = 0.0
        self.trajectory_points[0]["arc_length"] = 0.0
        self.trajectory_points[0]["time"] = 0.0
        self.trajectory_points[0]["psi_dot"] = 0.0
        self.trajectory_points[0]["bank"] = 0.0

        for i in range(1, len(self.trajectory_points)):
            p0 = np.array(self.trajectory_points[i - 1]["pos"])
            p1 = np.array(self.trajectory_points[i]["pos"])
            segment_length = np.linalg.norm(p1 - p0)
            cumulative_length += segment_length

            self.trajectory_points[i]["arc_length"] = cumulative_length
            self.trajectory_points[i]["time"] = cumulative_length / velocity

            # Calculate heading rate (psi_dot)
            psi0 = self.trajectory_points[i - 1]["psi"]
            psi1 = self.trajectory_points[i]["psi"]

            # Handle angle wraparound
            dpsi = psi1 - psi0
            while dpsi > np.pi:
                dpsi -= 2 * np.pi
            while dpsi < -np.pi:
                dpsi += 2 * np.pi

            # Time between points
            dt = segment_length / velocity if segment_length > 0 else 1e-6
            psi_dot = dpsi / dt

            # Calculate required bank angle: phi = atan(V * psi_dot / g)
            # For coordinated turn: bank angle relates to turn rate
            # Negative sign because positive yaw rate (left turn) requires
            # negative roll (left bank)
            bank = -np.arctan2(velocity * psi_dot, g)

            self.trajectory_points[i]["psi_dot"] = psi_dot
            self.trajectory_points[i]["bank"] = bank

        # change to numpy array
        bank_angles = np.array([point["bank"] for point in self.trajectory_points])
        bank_angles = rolling_median_replace(bank_angles, thresh=0.1)

        for i, bank_angle in enumerate(bank_angles):
            self.trajectory_points[i]["bank"] = bank_angle

        self.total_trajectory_time = self.trajectory_points[-1]["time"]
        # self.get_logger().info(
        #     f'trajectory: {cumulative_length:.2f}m, {
        #         self.total_trajectory_time:.2f}s @ {velocity:.2f}m/s'
        # )

    def publish_reference_tf(self):
        """Publish TF frame at current reference position based on elapsed time."""
        if not self.trajectory_points:
            return

        current_time = self.get_clock().now()

        Kspeedup = 10
        if self.relative_pose_position[0] > 0:
            # if self.dist > self.get_parameter('max_distance').value:
            dt = (current_time - self.previous_time).nanoseconds
            self.elapsed_time += (
                abs(self.relative_pose_position[0]) / Kspeedup + 1
            ) * dt
        else:
            # behind
            if self.dist < self.get_parameter("max_distance").value:
                dt = (current_time - self.previous_time).nanoseconds
                self.elapsed_time += dt

        self.previous_time = current_time

        elapsed = self.elapsed_time / 1e9
        # elapsed = (current_time - self.start_time).nanoseconds / 1e9

        # Loop the trajectory
        trajectory_time = elapsed % self.total_trajectory_time

        # Find the point at this time using binary search or linear
        # interpolation
        ref_point = self._interpolate_trajectory(trajectory_time)

        if ref_point is None:
            return

        # Create and broadcast transform
        t = TransformStamped()
        t.header.stamp = current_time.to_msg()
        t.header.frame_id = self.racecourse.frame_id
        t.child_frame_id = self.get_parameter("reference_frame_id").value

        t.transform.translation.x = ref_point["pos"][0]
        t.transform.translation.y = ref_point["pos"][1]
        t.transform.translation.z = ref_point["pos"][2]

        # Convert orientation (roll, pitch, yaw) to quaternion
        roll = ref_point["bank"]
        pitch = 0.0
        yaw = ref_point["psi"]

        ref_speed = self.get_parameter("planner.velocity").value

        q = quaternion_from_euler(roll, pitch, yaw)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)

        pose = PoseWithCovarianceStamped()
        pose.header = t.header
        pose.pose.pose.position.x = ref_point["pos"][0]
        pose.pose.pose.position.y = ref_point["pos"][1]
        pose.pose.pose.position.z = ref_point["pos"][2]
        pose.pose.pose.orientation.x = q[0]
        pose.pose.pose.orientation.y = q[1]
        pose.pose.pose.orientation.z = q[2]
        pose.pose.pose.orientation.w = q[3]

        max_dist = self.get_parameter("max_distance").value
        pose.pose.covariance = [
            max_dist**2,
            0,
            0,
            0,
            0,
            0,
            0,
            max_dist**2,
            0,
            0,
            0,
            0,
            0,
            0,
            max_dist**2,
            0,
            0,
            0,
            0,
            0,
            0,
            0.1,
            0,
            0,
            0,
            0,
            0,
            0,
            0.1,
            0,
            0,
            0,
            0,
            0,
            0,
            0.1,
        ]

        self.pose_pub.publish(pose)

        speed = Twist()
        speed.linear.x = ref_speed
        self.speed_pub.publish(speed)

    def _interpolate_trajectory(self, time):
        """Interpolate trajectory point at given time."""
        # Find the two points that bracket this time
        if time <= 0:
            return self.trajectory_points[0]
        if time >= self.total_trajectory_time:
            return self.trajectory_points[-1]

        # Binary search for efficiency
        left, right = 0, len(self.trajectory_points) - 1
        while right - left > 1:
            mid = (left + right) // 2
            if self.trajectory_points[mid]["time"] < time:
                left = mid
            else:
                right = mid

        # Linear interpolation between left and right
        p0 = self.trajectory_points[left]
        p1 = self.trajectory_points[right]

        t0, t1 = p0["time"], p1["time"]
        alpha = (time - t0) / (t1 - t0) if t1 > t0 else 0.0

        # Interpolate position
        pos = [
            p0["pos"][0] + alpha * (p1["pos"][0] - p0["pos"][0]),
            p0["pos"][1] + alpha * (p1["pos"][1] - p0["pos"][1]),
            p0["pos"][2] + alpha * (p1["pos"][2] - p0["pos"][2]),
        ]

        # Interpolate heading (handle wraparound)
        psi0, psi1 = p0["psi"], p1["psi"]
        dpsi = psi1 - psi0
        # Wrap to [-pi, pi]
        while dpsi > np.pi:
            dpsi -= 2 * np.pi
        while dpsi < -np.pi:
            dpsi += 2 * np.pi
        psi = psi0 + alpha * dpsi

        # Interpolate bank angle
        bank = p0["bank"] + alpha * (p1["bank"] - p0["bank"])

        return {"pos": pos, "psi": psi, "bank": bank}

    def publish_visualization(self):
        frame = self.racecourse.frame_id
        stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()
        mid = 0

        factory = MarkerFactory(frame)

        for seg_idx, seg in enumerate(self.segments):
            if self.get_parameter("show_path_marker").value:
                mid = self._add_path_marker(markers, factory, stamp, seg, seg_idx, mid)

            if self.get_parameter("show_headings").value:
                mid = self._add_heading_arrows(markers, factory, stamp, seg, mid)

            if self.get_parameter("show_turn_circles").value:
                mid = self._add_segment_turn_circles(
                    markers, factory, stamp, seg, seg_idx, mid
                )

        if self.get_parameter("show_gate_turn_circles").value:
            mid = self._add_gate_turn_circles(markers, factory, stamp, mid)

        self.marker_pub.publish(markers)
        self._publish_path(frame, stamp)

    def _add_path_marker(self, markers, factory, stamp, seg, seg_idx, mid):
        points = [
            Point(x=p["pos"][0], y=p["pos"][1], z=p["pos"][2]) for p in seg["points"]
        ]
        m = factory.line_strip(
            marker_id=mid,
            points=points,
            stamp=stamp,
            ns=f"path_{seg_idx}",
            line_width=0.15,
            color=factory.color(0.0, 0.5, 1.0),
        )
        markers.markers.append(m)
        return mid + 1

    def _add_heading_arrows(self, markers, factory, stamp, seg, mid):
        spacing = self.get_parameter("planner.heading_spacing").value
        for i in range(0, len(seg["points"]), spacing):
            pt = seg["points"][i]
            m = factory.arrow_pose(
                marker_id=mid,
                position=pt["pos"],
                heading=pt["psi"],
                stamp=stamp,
                ns="headings",
                length=1.0,
                width=0.15,
                color=factory.color(0.0, 1.0, 0.0, 0.8),
            )
            markers.markers.append(m)
            mid += 1
        return mid

    def _add_segment_turn_circles(self, markers, factory, stamp, seg, seg_idx, mid):
        R = self.get_parameter("turn_radius").value
        for cidx, c in enumerate(seg["centers"]):
            m = factory.circle(
                marker_id=mid,
                center=c[:2],
                radius=R,
                z=seg["z"],
                stamp=stamp,
                ns=f"circle_{seg_idx}_{cidx}",
                line_width=0.1,
                color=factory.color(0.2, 0.2, 0.2, 1.0),
            )
            markers.markers.append(m)
            mid += 1
        return mid

    def _add_gate_turn_circles(self, markers, factory, stamp, mid):
        R = self.get_parameter("turn_radius").value
        for gate in self.racecourse.gates:
            yaw = gate.yaw
            cx_right = gate.center[0] - R * np.sin(yaw)
            cy_right = gate.center[1] + R * np.cos(yaw)
            cx_left = gate.center[0] + R * np.sin(yaw)
            cy_left = gate.center[1] - R * np.cos(yaw)

            for label, cx, cy in [("L", cx_left, cy_left), ("R", cx_right, cy_right)]:
                m = factory.circle(
                    marker_id=mid,
                    center=[cx, cy],
                    radius=R,
                    z=gate.center[2],
                    stamp=stamp,
                    ns=f"gate_circle_{gate.name}_{label}",
                    line_width=0.05,
                    color=factory.color(1.0, 0.5, 0.0, 0.4),
                )
                markers.markers.append(m)
                mid += 1
        return mid

    def _publish_path(self, frame, stamp):
        factory = MarkerFactory(frame)
        path = Path()
        path.header = factory.header(stamp)
        for seg in self.segments:
            for pt in seg["points"]:
                pose = PoseStamped()
                pose.header = path.header
                pose.pose = factory.pose(pt["pos"], pt["psi"])
                path.poses.append(pose)
        self.path_pub.publish(path)


def main(args=None):
    rclpy.init(args=args)

    dubins_planner = DubinsGatePlannerNode()

    rclpy.spin(dubins_planner)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    dubins_planner.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
