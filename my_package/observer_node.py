#!/usr/bin/env python3
#
# This file is part of CyberShip Enterpries Suite.
#
# CyberShip Enterpries Suite software is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CyberShip Enterpries Suite is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# CyberShip Enterpries Suite. If not, see <https://www.gnu.org/licenses/>.
#
# Maintainer: Emir Cem Gezer
# Email: emir.cem.gezer@ntnu.no, emircem.gezer@gmail.com, me@emircem.com
# Year: 2022
# Copyright (C) 2024 NTNU Marine Cybernetics Laboratory

import rclpy
import rclpy.node
import numpy as np
import rcl_interfaces.msg

import std_msgs.msg
import tmr4243_interfaces.msg

from helpers.luenberger import luenberger
from helpers.dead_reckoning import dead_reckoning
from template_observer.wrap import wrap

from std_msgs.msg import Float64


class Observer(rclpy.node.Node):
    TASK_DEADRECKONING = 'deadreckoning'
    TASK_LUENBERGER = 'luenberger'
    TASK_LIST = [TASK_DEADRECKONING, TASK_LUENBERGER]

    def __init__(self):
        super().__init__('cse_observer')

        self.subs = {}
        self.pubs = {}

        self.measurement_lost = True

        # Subscribers for measured states
        self.subs["tau"] = self.create_subscription(
            std_msgs.msg.Float32MultiArray, '/tmr4243/state/tau', self.tau_callback, 10
        )
        self.subs["eta"] = self.create_subscription(
            std_msgs.msg.Float32MultiArray, '/tmr4243/state/eta', self.eta_callback, 10
        )

        # Publishers for observer estimates and error
        self.pubs['observer'] = self.create_publisher(
            tmr4243_interfaces.msg.Observer, '/tmr4243/observer/eta', 1
        )
        self.pubs['observer_error'] = self.create_publisher(
            Float64, "/observer_error", 10
        )

        # Parameters for observer gains
        self.task = Observer.TASK_LUENBERGER
        self.declare_parameter(
            'task',
            self.task,
            rcl_interfaces.msg.ParameterDescriptor(
                description="Task",
                type=rcl_interfaces.msg.ParameterType.PARAMETER_STRING,
                read_only=False,
                additional_constraints=f"Allowed values: {' '.join(Observer.TASK_LIST)}"
            )
        )
        self.task = self.get_parameter('task').get_parameter_value().string_value

        self.L1_value = [10.0] * 3
        self.declare_parameter(
            'L1',
            self.L1_value,
            rcl_interfaces.msg.ParameterDescriptor(
                description="L1 gain",
                type=rcl_interfaces.msg.ParameterType.PARAMETER_DOUBLE_ARRAY,
                read_only=False
            )
        )

        self.L2_value = [1.0] * 3
        self.declare_parameter(
            'L2',
            self.L2_value,
            rcl_interfaces.msg.ParameterDescriptor(
                description="L2 gain",
                type=rcl_interfaces.msg.ParameterType.PARAMETER_DOUBLE_ARRAY,
                read_only=False
            )
        )

        self.L3_value = [0.1] * 3
        self.declare_parameter(
            'L3',
            self.L3_value,
            rcl_interfaces.msg.ParameterDescriptor(
                description="L3 gain",
                type=rcl_interfaces.msg.ParameterType.PARAMETER_DOUBLE_ARRAY,
                read_only=False
            )
        )

        self.L1_value = self.get_parameter('L1').get_parameter_value().double_array_value
        self.L2_value = self.get_parameter('L2').get_parameter_value().double_array_value
        self.L3_value = self.get_parameter('L3').get_parameter_value().double_array_value

        self.get_logger().info(f"Task: {self.task}")

        # Initialize state estimates and measurements
        self.last_eta = None  # Measured eta (3x1)
        self.last_tau = None  # Measured tau (3x1)
        self.x_hat = None  # Full state estimate [eta_hat; nu_hat; bias_hat] (9x1)
        self.eta_hat = None  # Estimated eta (3x1)

        # Timer for observer loop
        self.observer_runner = self.create_timer(0.1, self.observer_loop)

    def observer_loop(self):
        # Wait until state is initialized and measurements are received
        if self.x_hat is None or self.last_eta is None or self.last_tau is None:
            self.get_logger().warn("Waiting for initial measurements to initialize state estimate.")
            return

        # Compute observer gains
        L1 = np.diag(self.L1_value)
        L2 = np.diag(self.L2_value)
        L3 = np.diag(self.L3_value)

        if self.measurement_lost:
            # Use pure dead-reckoning (no measurement correction)
            eta_hat, nu_hat, bias_hat = dead_reckoning(self.x_hat, self.last_tau)

            # Use this pseudo-measurement in the observer
            # eta_hat, nu_hat, bias_hat = luenberger(self.x_hat, eta_hat, self.last_tau, L1, L2, L3)
        else:
            # Use full Luenberger observer with measurements
            eta_hat, nu_hat, bias_hat = luenberger(self.x_hat, self.last_eta, self.last_tau, L1, L2, L3)
        # Store estimated eta for error calculation
        self.eta_hat = eta_hat

        # Publish observer estimates
        obs = tmr4243_interfaces.msg.Observer()
        obs.eta = eta_hat.flatten().tolist()
        obs.nu = nu_hat.flatten().tolist()
        obs.bias = bias_hat.flatten().tolist()
        self.pubs['observer'].publish(obs)

        # Publish observer error
        self.observer_error_callback()

    def tau_callback(self, msg: std_msgs.msg.Float32MultiArray):
        self.last_tau = np.array([msg.data], dtype=float).T

    def eta_callback(self, msg: std_msgs.msg.Float32MultiArray):
        self.last_eta = np.array([msg.data], dtype=float).T
        # Initialize state estimate with first measurement
        if self.x_hat is None:
            self.x_hat = np.zeros((9, 1))
            self.x_hat[0:3] = self.last_eta
            self.get_logger().info("State estimator initialized with initial eta measurement.")

    def observer_error_callback(self):
        if self.eta_hat is None or self.last_eta is None:
            return

        # Calculate error between estimated and measured eta
        eta_error = np.linalg.norm(self.eta_hat - self.last_eta)  
        msg = Float64()
        msg.data = float(eta_error)
        self.pubs['observer_error'].publish(msg)
        self.get_logger().info(f'Observer error: {eta_error}, eta: {self.last_eta}, eta_hat: {self.eta_hat}')


def main():
    rclpy.init()
    node = Observer()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()