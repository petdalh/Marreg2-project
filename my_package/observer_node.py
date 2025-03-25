#!/usr/bin/env python3

import rclpy
import rclpy.node
import numpy as np
import std_msgs.msg
import tmr4243_interfaces.msg

from helpers.observers import luenberger
from helpers.observers import dead_reckoning
from std_msgs.msg import Float64


class Observer(rclpy.node.Node):
    def __init__(self):
        super().__init__('cse_observer')

        self.subs = {}
        self.pubs = {}
        self.measurement_lost = False

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

        # Hardcoded observer gains
        self.L1_value = [10.0, 10.0, 10.0]  # Hardcoded L1 gains
        self.L2_value = [1.0, 1.0, 1.0]     # Hardcoded L2 gains  
        self.L3_value = [0.1, 0.1, 0.1]     # Hardcoded L3 gains

        # Initialize state estimates and measurements
        self.last_eta = None  # Measured eta (3x1)
        self.last_tau = None  # Measured tau (3x1)
        self.x_hat = None     # Full state estimate [eta_hat; nu_hat; bias_hat] (9x1)
        self.eta_hat = None   # Estimated eta (3x1)

        # Timer for observer loop
        self.observer_runner = self.create_timer(0.1, self.observer_loop)

    def observer_loop(self):
        # Wait until state is initialized and measurements are received
        if self.x_hat is None or self.last_eta is None or self.last_tau is None:
            self.get_logger().warn("Waiting for initial measurements to initialize state estimate.")
            return

        # Create diagonal gain matrices
        L1 = np.diag(self.L1_value)
        L2 = np.diag(self.L2_value)
        L3 = np.diag(self.L3_value)

        if self.measurement_lost:
            # Use pure dead-reckoning (no measurement correction)
            eta_hat, nu_hat, bias_hat = dead_reckoning(self.x_hat, self.last_tau)
        else:
            # Use full Luenberger observer with measurements
            eta_hat, nu_hat, bias_hat = luenberger(self.x_hat, self.last_eta, self.last_tau, L1, L2, L3)
        
        # Store estimated eta for error calculation
        self.eta_hat = eta_hat
        self.x_hat = np.vstack([eta_hat, nu_hat, bias_hat])

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