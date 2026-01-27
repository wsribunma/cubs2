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
"""Autolevel controller for aircraft - SAFE/AS3X style stabilized autopilot."""
from beartype import beartype
from cyecca.dynamics.explicit import explicit
from cyecca.dynamics.explicit import input_var
from cyecca.dynamics.explicit import Model
from cyecca.dynamics.explicit import output_var
from cyecca.dynamics.explicit import param
from cyecca.dynamics.explicit import state
import cyecca.sym as cy
import numpy as np


@explicit
class AutolevelController:
    """Unified autolevel controller with all states, inputs, params, outputs."""

    # === States ===
    i_p: float = state(1, 0.0, 'roll rate integral')
    i_q: float = state(1, 0.0, 'pitch rate integral')

    # === Inputs ===
    q: float = input_var(4, desc='quaternion [w,x,y,z]')
    omega: float = input_var(3, desc='angular velocity body frame (rad/s)')
    vel: float = input_var(3, desc='velocity earth frame ENU (m/s)')
    # Manual mode inputs (pass-through)
    ail_manual: float = input_var(desc='manual aileron (rad)')
    elev_manual: float = input_var(desc='manual elevator (rad)')
    rud_manual: float = input_var(desc='manual rudder (rad)')
    thr_manual: float = input_var(desc='manual throttle')
    mode: float = input_var(desc='mode: 0=manual, 1=stabilized')

    # === Parameters ===
    # Rate command sensitivity (how much stick deflection causes rate change)
    # Stick-to-rate scaling (rad/s per full stick)
    stick_rate_scale_phi: float = param(3.5 * np.pi, desc='stick to roll rate (rad/s)')
    stick_rate_scale_theta: float = param(2.0 * np.pi, desc='stick to pitch rate (rad/s)')
    
    # Outer loop: auto-level gains (smooth return to level when stick centered)
    Kp_phi: float = param(3.0, desc='P gain roll auto-level (per rad error)')
    Kp_theta: float = param(2.0, desc='P gain pitch auto-level (per rad error)')

    # Inner loop: rate tracking (gyro-based, fast stabilization)
    Kp_p: float = param(0.5, desc='P gain roll rate tracking')
    Ki_p: float = param(0.15, desc='I gain roll rate tracking')
    Kp_q: float = param(0.3, desc='P gain pitch rate tracking')
    Ki_q: float = param(0.1, desc='I gain pitch rate tracking')
    
    # Manual mode gyro damping (gentle - helps with oscillations)
    Kd_p_manual: float = param(0.15, desc='damping roll rate in manual mode')
    Kd_q_manual: float = param(0.15, desc='damping pitch rate in manual mode')

    # Yaw damping (passive)
    Kp_r: float = param(0.3, desc='P gain yaw rate damping')

    # Speed control
    Kp_speed: float = param(0.7, desc='P gain speed')
    speed_ref: float = param(20.0, desc='reference speed (m/s)')

    # Trim offsets
    trim_aileron: float = param(0.0, desc='aileron trim offset (rad)')
    trim_elevator: float = param(0.0, desc='elevator trim offset (rad)')
    trim_rudder: float = param(0.0, desc='rudder trim offset (rad)')

    # Limits
    phi_max: float = param(np.deg2rad(50), desc='max bank angle (rad)')
    theta_max: float = param(np.deg2rad(30), desc='max pitch angle (rad)')
    ail_min: float = param(-0.5, desc='ail min (rad)')
    ail_max: float = param(0.5, desc='ail max (rad)')
    elev_min: float = param(-0.5, desc='elev min (rad)')
    elev_max: float = param(0.5, desc='elev max (rad)')
    rud_min: float = param(-0.5, desc='rud min (rad)')
    rud_max: float = param(0.5, desc='rud max (rad)')
    thr_min: float = param(0.0, desc='thr min')
    thr_max: float = param(1.0, desc='thr max')

    # === Outputs ===
    ail: float = output_var(desc='aileron (rad)')
    elev: float = output_var(desc='elevator (rad)')
    rud: float = output_var(desc='rudder (rad)')
    thr: float = output_var(desc='throttle')


def _saturate(val, low, high):
    """Saturate value between low and high."""
    return cy.fmin(cy.fmax(val, low), high)


@beartype
def autolevel_controller() -> Model:
    """
    Create SAFE/AS3X style autolevel controller.

    A cascaded gyro-based stabilization system:
    - Inner loop: rate damping (fast, gyro-based)
    - Outer loop: attitude hold (slow, uses quaternion-derived φ/θ)

    Returns
    -------
    Model
        Autolevel controller model with integral states and control outputs

    """
    model = Model(AutolevelController)
    m = model.v  # Unified namespace

    # Extract quaternion and compute Euler angles
    qw = m.q.sym[0]
    qx = m.q.sym[1]
    qy = m.q.sym[2]
    qz = m.q.sym[3]

    # Roll (phi) - aerospace ZYX sequence
    phi = cy.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))

    # Pitch (theta)
    theta = cy.asin(2.0 * (qw * qy - qz * qx))

    # Extract angular rates
    p_meas = m.omega.sym[0]
    q_meas = m.omega.sym[1]
    r_meas = m.omega.sym[2]

    # Airspeed from velocity
    speed_meas = cy.norm_2(m.vel.sym)

    # SAFE/AS3X Style Stabilization:
    # In stabilized mode, stick commands RATE; autopilot auto-levels when stick centered
    # This is rate-based stabilization + automatic leveling, not attitude hold
    
    # Stick input to desired rate commands (pilot commands rate directly)
    p_cmd_stick = m.ail_manual.sym * m.stick_rate_scale_phi.sym
    q_cmd_stick = m.elev_manual.sym * m.stick_rate_scale_theta.sym
    
    # Auto-level: when stick is centered, command returns to level attitude
    # Proportional feedback from ACTUAL attitude error to drive back to level (0°)
    # This ensures wings-level return regardless of current bank/pitch
    p_cmd_level = -m.Kp_phi.sym * phi  # Roll error: always drives to phi=0 (wings-level)
    q_cmd_level = -m.Kp_theta.sym * theta  # Pitch error: always drives to theta=0 (level)
    
    # Apply angle limits to STICK commands in stabilized mode (only limit pilot input)
    # If near max bank angle, reduce stick authority to prevent exceeding limit
    # Margin factor: 0.9 = start limiting at 90% of max angle
    angle_margin = 0.9
    phi_saturation_factor = cy.fmin(1.0, (m.phi_max.sym - cy.fabs(phi)) / (m.phi_max.sym * (1.0 - angle_margin)))
    theta_saturation_factor = cy.fmin(1.0, (m.theta_max.sym - cy.fabs(theta)) / (m.theta_max.sym * (1.0 - angle_margin)))
    
    # Limit stick-commanded rates by saturation factor (soft limit on pilot input)
    p_cmd_stick_limited = p_cmd_stick * phi_saturation_factor
    q_cmd_stick_limited = q_cmd_stick * theta_saturation_factor
    
    # In stabilized mode: blend LIMITED stick rate commands with auto-level
    # Auto-level is NEVER saturated - it always tries to level wings
    # In manual mode: stick directly controls surfaces
    p_cmd = p_cmd_stick_limited * (1.0 - m.mode.sym) + (p_cmd_stick_limited + p_cmd_level) * m.mode.sym
    q_cmd = q_cmd_stick_limited * (1.0 - m.mode.sym) + (q_cmd_stick_limited + q_cmd_level) * m.mode.sym

    # Inner loop: rate tracking (gyro-based rate damping)
    # Always active in stabilized mode to track commanded rates
    e_p = (p_cmd - p_meas) * m.mode.sym  # Rate error (only in stabilized)
    e_q = (q_cmd - q_meas) * m.mode.sym  # Rate error (only in stabilized)
    
    # In manual mode, provide some gyro damping to improve handling
    e_p_manual = -m.Kp_p.sym * p_meas  # Damping proportional to rate
    e_q_manual = -m.Kp_q.sym * q_meas  # Damping proportional to rate

    # ODEs: Integral states for rate tracking (only in stabilized mode)
    model.ode(m.i_p, e_p * m.mode.sym)
    model.ode(m.i_q, e_q * m.mode.sym)

    # Stabilized mode control outputs (rate-based)
    # PID rate controller to track commanded rates
    ail_attitude_fb = -1.5 * phi  # Direct roll error feedback (only for roll leveling)
    
    ail_stabilized = _saturate(
        m.Kp_p.sym * e_p + m.Ki_p.sym * m.i_p.sym + ail_attitude_fb,
        m.ail_min.sym, m.ail_max.sym
    )
    elev_stabilized = _saturate(
        m.Kp_q.sym * e_q + m.Ki_q.sym * m.i_q.sym,
        m.elev_min.sym, m.elev_max.sym
    )
    
    # Manual mode outputs (direct pass-through + gentle gyro damping)
    ail_manual_out = m.ail_manual.sym + e_p_manual
    elev_manual_out = m.elev_manual.sym + e_q_manual
    rud_stabilized = _saturate(-m.Kp_r.sym * r_meas, m.rud_min.sym, m.rud_max.sym)
    e_speed = m.speed_ref.sym - speed_meas
    thr_stabilized = _saturate(m.Kp_speed.sym * e_speed, m.thr_min.sym, m.thr_max.sym)

    # Mode switch: 0 = manual (direct + damping), 1 = stabilized (rate-based auto-level)
    # Aileron/Elevator: switch between manual and stabilized
    ail_out = (
        _saturate(ail_manual_out, m.ail_min.sym, m.ail_max.sym) * (1.0 - m.mode.sym) +
        ail_stabilized * m.mode.sym
    ) + m.trim_aileron.sym
    
    elev_out = (
        _saturate(elev_manual_out, m.elev_min.sym, m.elev_max.sym) * (1.0 - m.mode.sym) +
        elev_stabilized * m.mode.sym
    ) + m.trim_elevator.sym
    
    # Rudder: yaw damping in both modes
    rud_out = (
        m.rud_manual.sym * (1.0 - m.mode.sym) + rud_stabilized * m.mode.sym
    ) + m.trim_rudder.sym
    
    # Throttle: pilot always controls throttle directly
    thr_out = m.thr_manual.sym

    # Define outputs
    model.output(m.ail, ail_out)
    model.output(m.elev, elev_out)
    model.output(m.rud, rud_out)
    model.output(m.thr, thr_out)

    model.build()
    return model


__all__ = [
    'autolevel_controller',
    'AutolevelController',
]
