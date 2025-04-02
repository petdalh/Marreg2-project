#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
from helpers.thruster_allocation_extended import thruster_allocation_extended

class ThrustAllocationNode(Node):
    def __init__(self):
        super().__init__('thrust_allocation_node')
        self.get_logger().info("Thrust Allocation Node is running")
        
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/tmr4243/command/tau',
            self.tau_callback,
            10)
        self.u_publisher = self.create_publisher(Float32MultiArray, '/tmr4243/command/u', 10)

    def tau_callback(self, msg):
        tau = np.array(msg.data)
        u = thruster_allocation_extended(tau)
        if u is None:
            u = np.zeros(5)  
        u_msg = Float32MultiArray()
        u_msg.data = u.tolist()
        self.u_publisher.publish(u_msg)
        self.get_logger().info(f"Published u: {u}")

def main(args=None):
    rclpy.init(args=args)
    node = ThrustAllocationNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()