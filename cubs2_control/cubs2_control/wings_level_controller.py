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
"""Wings-level inner loop stabilization controller."""
from beartype import beartype
from cyecca.dynamics.explicit import explicit
from cyecca.dynamics.explicit import input_var
from cyecca.dynamics.explicit import Model
from cyecca.dynamics.explicit import output_var
from cyecca.dynamics.explicit import param
import cyecca.sym as cy
import numpy as np


@explicit
class WingsLevelController:
    """Inner loop wings-level stabilization - keeps aircraft wings level."""

    # === Inputs ===
    q: float = input_var(4, desc='quaternion [w,x,y,z]')
    omega: float = input_var(3, desc='angular velocity body frame (rad/s)')
    ail_manual: float = input_var(desc='manual aileron (rad)')
    elev_manual: float = input_var(desc='manual elevator (rad)')
    rud_manual: float = input_var(desc='manual rudder (rad)')
    thr_manual: float = input_var(desc='manual throttle')

    # === Parameters ===
    # Roll stabilization (P control on roll angle error)
    Kp_phi: float = param(2.0, desc='P gain roll angle control')
    
    # Manual mode gyro damping (gentle - helps with oscillations)
    Kp_p: float = param(0.15, desc='damping roll rate in manual mode')
    Kp_q: float = param(0.15, desc='damping pitch rate in manual mode')

    # Yaw damping (passive)
    Kp_r: float = param(0.3, desc='P gain yaw rate damping')

    # Trim offsets
    trim_aileron: float = param(0.0, desc='aileron trim offset (rad)')
    trim_elevator: float = param(0.0, desc='elevator trim offset (rad)')
    trim_rudder: float = param(0.0, desc='rudder trim offset (rad)')

    # Limits
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
def wings_level_controller() -> Model:
    """
    Create wings-level inner loop stabilization controller.
    
    Keeps aircraft wings level by applying proportional feedback on roll angle.
    Pitch and throttle are stick pass-through.

    Returns
    -------
    Model
        Wings-level controller model

    """
    model = Model(WingsLevelController)
    m = model.v  # Unified namespace

    # Extract quaternion and compute Euler angles
    qw = m.q.sym[0]
    qx = m.q.sym[1]
    qy = m.q.sym[2]
    qz = m.q.sym[3]

    # Roll (phi) - aerospace ZYX sequence
    phi = cy.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))

    # Extract angular rates
    p_meas = m.omega.sym[0]
    q_meas = m.omega.sym[1]
    r_meas = m.omega.sym[2]

    # Roll angle error from level (phi = 0 is wings level)
    phi_error = phi

    # Aileron: P controller on roll angle error to bring wings to level
    ail_stabilized = _saturate(
        -m.Kp_phi.sym * phi_error,  # Negative feedback: positive phi error → negative aileron
        m.ail_min.sym, m.ail_max.sym
    )

    # Elevator: stick pass-through with rate damping
    e_q_manual = -m.Kp_q.sym * q_meas  # Gentle gyro damping
    elev_manual_out = m.elev_manual.sym + e_q_manual
    elev_out = _saturate(elev_manual_out, m.elev_min.sym, m.elev_max.sym) + m.trim_elevator.sym

    # Rudder: passive yaw damping
    rud_stabilized = _saturate(-m.Kp_r.sym * r_meas, m.rud_min.sym, m.rud_max.sym)

    # Throttle: pilot always controls throttle directly
    thr_out = m.thr_manual.sym

    # Define outputs
    model.output(m.ail, ail_stabilized + m.trim_aileron.sym)
    model.output(m.elev, elev_out)
    model.output(m.rud, rud_stabilized + m.trim_rudder.sym)
    model.output(m.thr, thr_out)

    model.build()
    return model


__all__ = [
    'wings_level_controller',
    'WingsLevelController',
]
