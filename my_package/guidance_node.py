#!/usr/bin/env python3

import rclpy
import rclpy.node
import rcl_interfaces.msg
import tmr4243_interfaces.msg
import numpy as np

from helpers.stationkeeping import stationkeeping
from helpers.straight_line import straight_line, update_law


class Guidance(rclpy.node.Node):
    TASK_STATIONKEEPING = 'stationkeeping'
    TASK_STRAIGHT_LINE = 'straight_line'
    TASKS = [TASK_STRAIGHT_LINE, TASK_STATIONKEEPING]

    def __init__(self):
        super().__init__("tmr4243_guidance")

        self.pubs = {}
        self.subs = {}

        self.pubs["reference"] = self.create_publisher(
            tmr4243_interfaces.msg.Reference, '/tmr4243/control/reference', 1)

        self.subs["observed_eta"] = self.create_subscription(
            tmr4243_interfaces.msg.Observer, '/tmr4243/observer/x_hat', self.observer_callback, 10)

        self.last_observation = None

        self.declare_parameter('task', self.TASK_STRAIGHT_LINE)
        self.task = self.get_parameter('task').value

        # p0 = np.array([0.0, 0.0])
        # p1 = np.array([1.0, 0.0])
        # U_ref = 0.1
        # mu = 0.1
        # eps = 1e-6

        # Guidance parameters

        # Stationkeeping
        self.declare_parameter('eta_d', [0.0, 0.0, 0.0])
        self.eta_d = np.array(self.get_parameter('eta_d').value)

        # Straight Line
        # self.declare_parameter('p0', [0.0, 0.0])
        # self.declare_parameter('p1', [0.0, 1.0])
        self.declare_parameter('line_length', 4)
        self.declare_parameter('U_ref', 0.1)
        self.declare_parameter('mu', 0.1)
        self.declare_parameter('eps', 1e-6)

        # self.p0 = np.array(self.get_parameter('p0').value)
        # self.p1 = np.array(self.get_parameter('p1').value)
        self.line_length = self.get_parameter('line_length').value
        self.U_ref = self.get_parameter('U_ref').value
        self.mu = self.get_parameter('mu').value
        self.eps = self.get_parameter('eps').value

        self.p0 = np.array([0.0, 0.0])
        self.p1 = np.array([0.0, 0.0])
        # eta_d = np.array([0, 0, 0], dtype=float)

        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        guidance_period = 0.1  # seconds
        self.guidance_timer = self.create_timer(
            guidance_period, self.guidance_callback)

        self.dt = 0.1
        self.s = 0.0

        self.switched_to_stationkeeping = False  
        self.initialize = False
        self.station_keeping_eta = np.zeros(3)

    def timer_callback(self):

        self.task = self.get_parameter(
            'task').get_parameter_value().string_value

        # self.get_logger().info(f"Parameter task: {self.task}", throttle_duration_sec=1.0)

    def guidance_callback(self):

        if Guidance.TASK_STATIONKEEPING in self.task:
            eta_d, eta_ds, eta_ds2 = stationkeeping(self.eta_d)

            msg = tmr4243_interfaces.msg.Reference()
            msg.eta_d = eta_d.flatten().tolist()
            msg.eta_ds = eta_ds.flatten().tolist()
            msg.eta_ds2 = eta_ds2.flatten().tolist()

            msg.w = 0.0
            msg.v_s = 0.0
            msg.v_ss = 0.0

            self.get_logger().info(
                f"(Stationkeeping) sending eta_d: {msg.eta_d}")
            self.pubs["reference"].publish(msg)

            self.initial_setup = True

    def guidance_callback(self):
        if Guidance.TASK_STATIONKEEPING in self.task:
            eta_d, eta_ds, eta_ds2 = stationkeeping(self.eta_d)

            msg = tmr4243_interfaces.msg.Reference()
            msg.eta_d = eta_d.flatten().tolist()
            msg.eta_ds = eta_ds.flatten().tolist()
            msg.eta_ds2 = eta_ds2.flatten().tolist()

            msg.w = 0.0
            msg.v_s = 0.0
            msg.v_ss = 0.0

            self.get_logger().info(
                f"(Stationkeeping) sending eta_d: {msg.eta_d}")
            self.pubs["reference"].publish(msg)

        elif Guidance.TASK_STRAIGHT_LINE in self.task:
            if self.last_observation is None:
                self.get_logger().warn("Observer data is not available yet")
                return
            
            if not self.initialize:
                self.get_logger().info("Initialize straight line")
                if self.last_observation and self.last_observation.eta:
                    self.p0 = np.array([self.last_observation.eta[0], self.last_observation.eta[1]])
                    self.p1 = self.p0.copy()
                    
                    posx = self.p0[0] + self.line_length * np.cos(self.last_observation.eta[2])
                    posy = self.p0[1] + self.line_length * np.sin(self.last_observation.eta[2])
                    self.p1 = np.array([posx, posy])

                    self.initialize = True
                    self.switched_to_stationkeeping = False  # Reset switch flag
                    self.get_logger().info(f"Initialized p0: {self.p0}, p1: {self.p1}")
                else:
                    self.get_logger().warn("Cannot initialize line: last_observation or eta is missing.")
                    return

            # Check if we need to switch to stationkeeping
            if np.abs(self.s) >= 1:
                # Save current position when we first switch to stationkeeping
                if not self.switched_to_stationkeeping:
                    self.station_keeping_eta = np.array(self.last_observation.eta)
                    self.switched_to_stationkeeping = True
                    self.get_logger().info(f"Switching to stationkeeping at position: {self.station_keeping_eta}")

                # Perform stationkeeping at the saved position
                eta_d, eta_ds, eta_ds2 = stationkeeping(self.station_keeping_eta)
                
                msg = tmr4243_interfaces.msg.Reference()
                msg.eta_d = eta_d.flatten().tolist()
                msg.eta_ds = eta_ds.flatten().tolist()
                msg.eta_ds2 = eta_ds2.flatten().tolist()
                msg.w = 0.0
                msg.v_s = 0.0
                msg.v_ss = 0.0

                self.pubs["reference"].publish(msg)
                self.get_logger().info(f"(Stationkeeping) sending eta_d: {msg.eta_d}")
                return

            # Continue with straight line guidance if s < 1
            w, v_s, v_ss = update_law(
                self.last_observation, self.s, self.eps, self.p0, self.p1, self.U_ref, self.mu)
            self.s += (w + v_s) * self.dt
            self.get_logger().info(f"s: {self.s}")
            
            eta_d, eta_ds, eta_ds2 = straight_line(
                self.s, self.p0, self.p1, self.U_ref, self.mu)

            msg = tmr4243_interfaces.msg.Reference()
            msg.eta_d = eta_d.flatten().tolist()
            msg.eta_ds = eta_ds.flatten().tolist()
            msg.eta_ds2 = eta_ds2.flatten().tolist()
            msg.w = w
            msg.v_s = v_s
            msg.v_ss = v_ss

            self.pubs["reference"].publish(msg)
            self.get_logger().info(f"(Straight Line) sending eta_d: {msg.eta_d}")

    def observer_callback(self, msg):
        self.last_observation = msg


def main(args=None):
    # Initialize the node
    rclpy.init(args=args)

    node = Guidance()

    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
