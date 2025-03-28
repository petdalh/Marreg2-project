#!/usr/bin/env python3

import rclpy
import rclpy.node
import numpy as np
import std_msgs.msg
import tmr4243_interfaces.msg
from helpers.observers import luenberger
from helpers.observers import dead_reckoning
from std_msgs.msg import Float64
from sensor_msgs.msg import Joy

class Observer(rclpy.node.Node):
    def __init__(self):
        super().__init__('cse_observer')
        self.subs = {}
        self.pubs = {}
        self.measurement_lost = False
        self.sample_count = 0
        self.max_missing_samples = 4 
        self.button_was_pressed = False
        self.manual_override = False 

        self.subs["tau"] = self.create_subscription(
            std_msgs.msg.Float32MultiArray, '/tmr4243/state/tau', self.tau_callback, 10
        )
        self.subs["eta"] = self.create_subscription(
            std_msgs.msg.Float32MultiArray, '/tmr4243/state/eta', self.eta_callback, 10
        )
        self.subs["joy"] = self.create_subscription(
            Joy, '/joy', self.joy_callback, 10
        )

        self.pubs['observer'] = self.create_publisher(
            tmr4243_interfaces.msg.Observer, '/tmr4243/observer/x_hat', 1
        )
        self.pubs['observer_error'] = self.create_publisher(
            Float64, "/observer_error", 10
        )

        self.L1_value = [10.0, 10.0, 10.0]
        self.L2_value = [1.0, 1.0, 1.0]  
        self.L3_value = [0.1, 0.1, 0.1]

        self.last_eta = None
        self.last_tau = None
        self.x_hat = None
        self.eta_hat = None

        self.observer_runner = self.create_timer(0.1, self.observer_loop)

    def observer_loop(self):
        if self.x_hat is None or self.last_eta is None or self.last_tau is None:
            self.get_logger().warn("Waiting for initial measurements to initialize state estimate.")
            return

        L1 = np.diag(self.L1_value)
        L2 = np.diag(self.L2_value)
        L3 = np.diag(self.L3_value)

        if self.last_eta is not None:
            self.sample_count += 1
            if self.sample_count >= self.max_missing_samples and not self.manual_override:
                self.measurement_lost = True
                self.get_logger().warn(f"Measurement signal lost - no updates for {self.sample_count} samples")
        else:
            self.measurement_lost = True

        if self.measurement_lost:
            eta_hat, nu_hat, bias_hat = dead_reckoning(self.x_hat, self.last_tau)
            self.get_logger().info("Using dead-reckoning", throttle_duration_sec=1.0)
        else:
            eta_hat, nu_hat, bias_hat = luenberger(self.x_hat, self.last_eta, self.last_tau, L1, L2, L3)
        
        self.eta_hat = eta_hat
        self.x_hat = np.vstack([eta_hat, nu_hat, bias_hat])

        obs = tmr4243_interfaces.msg.Observer()
        obs.eta = eta_hat.flatten().tolist()
        obs.nu = nu_hat.flatten().tolist()
        obs.bias = bias_hat.flatten().tolist()
        self.pubs['observer'].publish(obs)

        self.observer_error_callback()

    def tau_callback(self, msg: std_msgs.msg.Float32MultiArray):
        self.last_tau = np.array([msg.data], dtype=float).T

    def eta_callback(self, msg: std_msgs.msg.Float32MultiArray):
        self.last_eta = np.array([msg.data], dtype=float).T
        self.sample_count = 0
        if not self.manual_override:
            self.measurement_lost = False
        if self.x_hat is None:
            self.x_hat = np.zeros((9, 1))
            self.x_hat[0:3] = self.last_eta
            self.get_logger().info("State estimator initialized with initial eta measurement.")

    def joy_callback(self, msg: Joy):
        self.latest_axes = msg.axes
        if len(msg.buttons) > 0:
            button_is_pressed = (msg.buttons[0] == 1)
            if button_is_pressed and not self.button_was_pressed:
                self.measurement_lost = not self.measurement_lost
                self.manual_override = self.measurement_lost
                self.get_logger().info(f"Measurement signal toggled to {'lost' if self.measurement_lost else 'active'}")
            self.button_was_pressed = button_is_pressed

    def observer_error_callback(self):
        if self.eta_hat is None or self.last_eta is None:
            return
        eta_error = np.linalg.norm(self.eta_hat - self.last_eta)
        msg = Float64()
        msg.data = float(eta_error)
        self.pubs['observer_error'].publish(msg)
        if self.measurement_lost:
            self.get_logger().info(f'Observer (Dead-Reckoning) error: {eta_error}, eta: {self.last_eta}, eta_hat: {self.eta_hat}')
        else:
            self.get_logger().info(f'Observer (Luenberger) error: {eta_error}, eta: {self.last_eta}, eta_hat: {self.eta_hat}')

def main():
    rclpy.init()
    node = Observer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
