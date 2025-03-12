#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

class MyNode(Node):
    def __init__(self):
            super().__init__("my_publisher")

            topic = "/number"

            self.number_publisher = self.create_publisher(Float64, topic, 10)

            msg = Float64()
            
            self.create_timer(0.5, self.timer_callback)

            self.get_logger().info("My publisher is running")

            
    
    def timer_callback(self):
         msg = Float64()
         msg.data = 10.0
         self.number_publisher.publish(msg)

if __name__ == "__main__":
    rclpy.init()
    node = MyNode()
    rclpy.spin(node)
    rclpy.shutdown()