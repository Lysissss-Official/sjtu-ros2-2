# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy, cv2
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PointStamped
import numpy as np
from std_msgs.msg import Bool, Int32, Float32MultiArray
import math
import time

# ===LiDAR START ===
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data


# ==== LiDAR END ====


# ==== CV Land Detection ====
class LaneDetection(Node):
    def __init__(self):
        super().__init__('lanedetection')
        self.get_logger().info("Start lane keeping.")

        self.point_sub = self.create_subscription(PointStamped, '/line_track_center_detection', self.point_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/lane_cmd_vel', 10)
        # self.obs_sub = self.create_subscription(Bool, '/has_obs',self.obs_callback,10)

        self.twist = Twist()
        self.point = PointStamped()
        # self.has_obs = 0

    def pixel_to_world(self, x0, y0):
        # 垂直角
        theta = math.radians(
            30.0 + (y0 - 540.0) * 0.02685
        )

        # 前方距离
        Y = 0.05 / math.tan(theta)

        # 水平角
        phi = math.radians(
            (x0 - 960.0) * 0.02604
        )

        # 横向距离
        X = Y * math.tan(phi)

        return X, Y

    def on_shutdown(self):
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.cmd_vel_pub.publish(self.twist)

        # def obs_callback(self, msg):

    #    self.has_obs = msg.data

    def point_callback(self, msg):
        self.point = msg

        x_img = msg.point.x
        y_img = msg.point.y

        # 图像坐标 -> 地面坐标
        X, Y = self.pixel_to_world(
            x_img,
            y_img
        )

        # 误差偏航角(rad)
        error = math.atan2(X, Y)
        val = 2.25

        # 前进速度
        self.twist.linear.x = 0.25 * val

        # P控制
        kp = 4.5 * val

        self.twist.angular.z = -kp * error
        self.cmd_vel_pub.publish(self.twist)


# ===== Lidar Obstacle Detection ====
class ViewLidar(Node):
    def __init__(self, name):
        super().__init__(name)
        self.get_logger().info("LiDAR is OK")
        self.laser_sub = self.create_subscription(LaserScan, "/scan", self.laser_callback, qos_profile_sensor_data)
        self.obs_pub = self.create_publisher(Bool, '/has_obstacle', 10)

    def laser_callback(self, scan_data):
        # Right: 125
        # Front: 250
        # Left:  375
        # Back:  500
        has_obstacle = False

        for i in range(max(0, 250 - 40), min(len(scan_data.ranges), 250 + 40)):
            if 0.12 < scan_data.ranges[i] < 0.3:
                has_obstacle = True
                break
        msg = Bool()
        msg.data = has_obstacle

        self.obs_pub.publish(msg)


# ==== Action Arbiter ====
class Combination(Node):
    STATE_STOP = "STOP"
    STATE_FOLLOW = "FOLLOW"
    STATE_SLOW = "SLOW"
    STATE_TURN = "TURN"

    def __init__(self):
        super().__init__('combination')

        self.obstacle = False
        self.lane_twist = Twist()

        # 避障状态
        self.state = "STOP"
        self.time_start = time.time()

        self.time_start_one = time.time()

        # 交通标志运动控制
        self.running = False

        self.sign_candidate = -1
        self.sign_count = 0
        self.last_sign = -1

        self.have_used_3 = False

        self.raw_scores = [0.0, 0.0, 0.0, 0.0]

        self.lane_sub = self.create_subscription(
            Twist,
            '/lane_cmd_vel',
            self.lane_callback,
            10
        )

        self.obs_sub = self.create_subscription(
            Bool,
            '/has_obstacle',
            self.obs_callback,
            10
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.sign_sub = self.create_subscription(
            Int32,
            '/traffic_sign_class',
            self.sign_callback,
            10
        )

        self.scores_sub = self.create_subscription(
            Float32MultiArray,
            '/traffic_sign_scores',
            self.scores_callback,
            10
        )

        self.timer = self.create_timer(
            0.01667,
            self.Priority
        )

    def lane_callback(self, msg):
        self.lane_twist = msg

    def obs_callback(self, msg):
        self.obstacle = msg.data

    def sign_callback(self, msg):

        sign_id = int(msg.data)

        if sign_id == -1 and self.raw_scores[3] < 0.3:
            if self.have_used_3 == False and time.time() - self.time_start_one >= 13:
                self.have_used_3 = True
                self.candicate = 3
                self.sign_count = 2
                self.state = "SLOW"
                self.time_start = time.time()
                return

            elif self.last_sign == 1:
                if sign_id == self.sign_candidate:
                    self.sign_count += 1
                else:
                    self.sign_candidate = sign_id
                    self.sign_count = 1

                if self.sign_count >= 5:
                    self.state = "FOLLOW"
                    self.time_start = time.time()
                    self.time_start_one = time.time()
                    self.last_sign = -1
                return

            else:
                return

        # print(time.time()-self.time_start_one)

        if self.raw_scores[3] >= 0.3:
            sign_id = 3

        if sign_id == self.sign_candidate:
            self.sign_count += 1
        else:
            self.sign_candidate = sign_id
            self.sign_count = 1

        # if self.raw_scores[3]>=0.25:
        #    self.sign_count = 2

        # 两帧确认
        if self.sign_count < 2:
            return

        # 防止连续触发
        if sign_id == self.last_sign:
            return

        self.last_sign = sign_id

        if sign_id == 1:
            self.state = "STOP"
            self.time_start = time.time()

        elif sign_id == 2:
            self.state = "STOP"
            self.time_start = time.time()

        elif sign_id == 3:
            self.state = "SLOW"
            self.time_start = time.time()
        print(sign_id, "\n")

    def scores_callback(self, msg):
        # print("raw scores:", list(msg.data))
        self.raw_scores = list(msg.data)

    def Priority(self):
        cmd = Twist()
        if self.state == "STOP":
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        if self.state == "FOLLOW":

            if self.obstacle:
                self.state = "TURN"
                self.time_start = time.time()

            else:
                cmd = self.lane_twist

        elif self.state == "SLOW":

            # 减速期间仍然循迹
            cmd = self.lane_twist
            cmd.linear.x = cmd.linear.x * 0.6

            if self.obstacle:
                self.state = "TURN"
                self.time_start = time.time()

            elif time.time() - self.time_start > 1.1:
                self.state = "FOLLOW"
                self.time_start = time.time()



        elif self.state == "TURN":
            wait_time = 0.2

            if time.time() - self.time_start <= wait_time:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
            elif time.time() - self.time_start > wait_time:
                cmd.linear.x = 0.0
                cmd.angular.z = 2.5

            if time.time() - self.time_start > wait_time + 1.4:
                self.state = "FOLLOW"

        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)

    lanedetection = LaneDetection()
    lidarview = ViewLidar("View_YDLidar")
    combination = Combination()

    executor = MultiThreadedExecutor()

    executor.add_node(lanedetection)
    executor.add_node(lidarview)
    executor.add_node(combination)

    try:
        executor.spin()

    except KeyboardInterrupt:
        pass

    finally:
        executor.destroy_node(combination)
        executor.destroy_node(lidarview)
        executor.destroy_node(lanedetection)
        rclpy.shutdown()


if __name__ == '__main__':
    main()
