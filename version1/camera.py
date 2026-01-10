#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

import cv2
import mediapipe as mp
import numpy as np
import time


class HandControlNode(Node):

    def __init__(self):
        super().__init__('hand_control_node')

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Cannot open camera")
            exit()

        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Timer (30 Hz)
        self.timer = self.create_timer(0.03, self.timer_callback)

        # Safety
        self.last_hand_time = time.time()
        self.timeout_sec = 0.5

        self.get_logger().info("Hand control node started")


    def count_fingers(self, hand_landmarks):
        """
        Counts extended fingers (ignores thumb)
        """
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]

        count = 0
        for tip, pip in zip(finger_tips, finger_pips):
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
                count += 1
        return count


    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        twist = Twist()
        hand_detected = False

        if result.multi_hand_landmarks:
            hand_detected = True
            self.last_hand_time = time.time()

            hand_landmarks = result.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )

            # Count fingers
            finger_count = self.count_fingers(hand_landmarks)

            # Palm center (wrist landmark)
            palm_x = hand_landmarks.landmark[0].x  # normalized 0–1

            # Gesture logic
            if finger_count >= 4:
                gesture = "OPEN HAND"
                twist.linear.x = 0.2
            elif finger_count == 0:
                gesture = "FIST"
                twist.linear.x = 0.0
            else:
                gesture = "UNKNOWN"
                twist.linear.x = 0.0

            # Steering logic
            if palm_x < 0.4:
                twist.angular.z = 0.6
            elif palm_x > 0.6:
                twist.angular.z = -0.6
            else:
                twist.angular.z = 0.0

            # Display info
            cv2.putText(
                frame,
                f"Gesture: {gesture}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        # Safety stop if hand disappears
        if not hand_detected:
            if time.time() - self.last_hand_time > self.timeout_sec:
                twist = Twist()

        self.cmd_pub.publish(twist)

        cv2.imshow("Hand Control", frame)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = HandControlNode()
    rclpy.spin(node)

    node.cap.release()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
