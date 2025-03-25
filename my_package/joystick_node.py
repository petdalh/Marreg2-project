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
        
        # Store the latest values from callbacks
        self.current_eta = np.zeros(3)
        self.latest_axes = None
        self.Basin = False

        # Subscribers update only internal state
        self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.create_subscription(Float32MultiArray, '/tmr4243/state/eta', self.eta_callback, 10)
        
        # Publisher for thruster commands
        self.u_publisher = self.create_publisher(Float32MultiArray, '/tmr4243/command/u', 10)
        
        # Timer to process and publish commands
        self.timer = self.create_timer(0.1, self.timer_callback)

    def eta_callback(self, msg: Float32MultiArray):
        self.current_eta = np.array(msg.data)

    def joy_callback(self, msg: Joy):
        self.latest_axes = msg.axes

    def timer_callback(self):
        self.publish_u()

    def calculate_command_computation(self):
        # Only process if we have received joystick input
        if self.latest_axes is None:
            return

        # Process input and compute thruster commands
        tau_body = handle_controller_input(self.latest_axes)
        heading = self.current_eta[2] if len(self.current_eta) >= 3 else 0.0
        if self.Basin == True:
            R = create_R(heading)
            tau_basin = np.linalg.inv(R) @ tau_body
            u = thruster_allocation_extended(tau_basin).tolist()
        else:
            u = thruster_allocation_extended(tau_body).tolist()

        if u == None:
            return np.zeros(5)

        return u, heading

    def publish_u(self):
        u, heading = self.calculate_command_computation()
        # Publish commands
        u_message = Float32MultiArray()
        u_message.data = u
        self.u_publisher.publish(u_message)
        self.get_logger().info(f"Heading: {heading:.2f}, u calculated: {u}")

def main(args=None):
    rclpy.init(args=args)
    node = JoystickNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
