#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray
import numpy as np
from helpers.thruster_allocation_extended import thruster_allocation_extended
from helpers.create_R import create_R
from helpers.joystick_helpers import handle_controller_input

class JoystickNode(Node):
    def __init__(self):
        super().__init__('joystick_node')
        self.get_logger().info("My node is running")
        
        self.current_eta = np.zeros(3)
        self.latest_axes = None
        self.Basin = False

        self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.create_subscription(Float32MultiArray, '/tmr4243/state/eta', self.eta_callback, 10)
        
        self.u_publisher = self.create_publisher(Float32MultiArray, '/tmr4243/command/u', 10)
        self.tau_publisher = self.create_publisher(Float32MultiArray, '/tmr4243/command/tau', 10)
        
        self.timer = self.create_timer(0.1, self.timer_callback)

    def eta_callback(self, msg: Float32MultiArray):
        self.current_eta = np.array(msg.data)

    def joy_callback(self, msg: Joy):
        self.latest_axes = msg.axes

    def timer_callback(self):
        result = self.calculate_command_computation()
        if result is None:
            return
        u, heading, tau = result
        
        u_message = Float32MultiArray()
        u_message.data = u
        self.u_publisher.publish(u_message)
        
        tau_message = Float32MultiArray()
        tau_message.data = tau.tolist()  
        self.tau_publisher.publish(tau_message)
        
        self.get_logger().info(f"Heading: {heading:.2f}, u: {u}, tau: {tau.tolist()}")

    def calculate_command_computation(self):
        if self.latest_axes is None:
            return None

        tau_body = handle_controller_input(self.latest_axes)
        heading = self.current_eta[2] if len(self.current_eta) >= 3 else 0.0
        if self.Basin:
            R = create_R(heading)
            tau = np.linalg.inv(R) @ tau_body
        else:
            tau = tau_body

        u = thruster_allocation_extended(tau)
        if u is None:
            return np.zeros(5), heading, tau

        return u, heading, tau

def main(args=None):
    rclpy.init(args=args)
    node = JoystickNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
