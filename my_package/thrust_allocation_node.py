#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
from helpers.thruster_allocation_extended import thruster_allocation_extended

class ThrusterAllocationNode(Node):
    def __init__(self):
        super().__init__('thruster_allocation_node')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/tmr4243/command/tau',
            self.tau_callback,
            10
        )
        self.publisher = self.create_publisher(
            Float32MultiArray,
            '/tmr4243/command/u',
            10
        )

    def tau_callback(self, msg):
        tau = np.array(msg.data, dtype=np.float32).reshape(3, 1)
        u = thruster_allocation_extended(tau)
        
        if u is not None:
            u_msg = Float32MultiArray()
            u_msg.data = u.flatten().tolist()
            self.publisher.publish(u_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ThrusterAllocationNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()