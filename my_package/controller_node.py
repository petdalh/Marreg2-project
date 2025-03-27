#!/usr/bin/env python3

import rclpy
import rclpy.node
import rcl_interfaces.msg
import tmr4243_interfaces.msg
import std_msgs.msg
import numpy as np

from helpers.controllers import PID_controller


class Controller(rclpy.node.Node):
    TASK_PD_FF_CONTROLLER = 'PD_FF_controller'
    TASK_PID_CONTROLLER = 'PID_controller'
    TASK_BACKSTEPPING_CONTROLLER = 'backstepping_controller'
    TASKS = [TASK_PD_FF_CONTROLLER, TASK_PID_CONTROLLER, TASK_BACKSTEPPING_CONTROLLER]

    def __init__(self):
        super().__init__("tmr4243_controller")

        # Declare parameters with default values
        self.declare_parameter('task', self.TASK_PID_CONTROLLER)
        self.declare_parameter('p_gain', 1.0)
        self.declare_parameter('i_gain', 0.1)
        self.declare_parameter('d_gain', 0.5)
        
        # Get parameters
        self.task = self.get_parameter('task').value
        self.p_gain = self.get_parameter('p_gain').value
        self.i_gain = self.get_parameter('i_gain').value
        self.d_gain = self.get_parameter('d_gain').value

        self.pubs = {}
        self.subs = {}

        self.subs["reference"] = self.create_subscription(
            tmr4243_interfaces.msg.Reference, '/tmr4243/control/reference', self.received_reference, 10)

        self.subs['observer'] = self.create_subscription(
            tmr4243_interfaces.msg.Observer, '/tmr4243/observer/eta', self.received_observer ,10)

        self.pubs["tau_cmd"] = self.create_publisher(
            std_msgs.msg.Float32MultiArray, '/tmr4243/command/tau', 1)

        self.u_publisher = self.create_publisher(Float32MultiArray, '/tmr4243/command/u', 10)

        self.last_reference = None

        self.last_observation = None

        timer_period = 0.1 # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        controller_period = 0.1 # seconds
        self.controller_timer = self.create_timer(controller_period, self.controller_callback)


    def timer_callback(self):

        self.get_logger().info(f"Parameter task: {self.task}", throttle_duration_sec=1.0)

    def controller_callback(self):

        if self.last_reference == None or self.last_observation == None:
            self.get_logger().warn("Last reference or last observation is None", throttle_duration_sec=1.0)
            return

        if Controller.TASK_PD_FF_CONTROLLER in self.task:
            tau = PD_FF_controller(
                self.last_observation,
                self.last_reference,
                self.p_gain,
                self.d_gain
            )
        elif Controller.TASK_PID_CONTROLLER  in self.task:
            tau = PID_controller(
                self.last_observation,
                self.last_reference,
                self.p_gain,
                self.i_gain,
                self.d_gain
            )
        elif Controller.TASK_BACKSTEPPING_CONTROLLER in self.task:
            tau = backstepping_controller(
                self.last_observation,
                self.last_reference,
                self.k1_gain,
                self.k2_gain
            )
        else:
            tau = np.zeros((3, 1), dtype=float)

        if len(tau) != 3:
            self.get_logger().warn(f"tau has length of {len(tau)} but it should be 3: tau := [Fx, Fy, Mz]", throttle_duration_sec=1.0)
            return


        tau_cmd = std_msgs.msg.Float32MultiArray()
        tau_cmd.data = tau.flatten().tolist()
        self.pubs["tau_cmd"].publish(tau_cmd)

    def received_reference(self, msg):
        self.last_reference = msg

    def received_observer(self, msg):
        self.last_observation = msg


def main(args=None):
    rclpy.init(args=args)

    rclpy.spin(Controller())

    rclpy.shutdown()

if __name__ == '__main__':
    main()