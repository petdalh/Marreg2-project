#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64


class MyNode(Node):
    def __init__(self):
        super().__init__("my_node")

        topic = "/number"

        self.create_subscription(Float64, topic, self.number_callback, 10)

        self.get_logger().info("My node is running, but now it can subscribe")

    def number_callback(self, msg: Float64):
        number = msg.data 
        self.get_logger().info(f"Recieved {number}")

if __name__ == "__main__": 
    rclpy.init()
    node = MyNode()
    rclpy.spin(node)
    rclpy.shutdown()









