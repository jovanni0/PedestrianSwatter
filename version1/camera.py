#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import cv2
import mediapipe as mp
import math
import numpy as np

class HandControlNode(Node):
    def __init__(self):
        super().__init__('hand_control_node')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.033, self.run_loop)

        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)
        self.mp_draw = mp.solutions.drawing_utils

        # Config
        self.sensitivity = 0.2 
        self.max_speed = 0.8
        self.deadzone_deg = 5.0

    def run_loop(self):
        success, frame = self.cap.read()
        if not success: return
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        result = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        twist = Twist()

        if result.multi_hand_landmarks:
            lms = result.multi_hand_landmarks[0].landmark
            self.mp_draw.draw_landmarks(frame, result.multi_hand_landmarks[0], self.mp_hands.HAND_CONNECTIONS)

            # 1. Wrist and Points
            wrist = lms[0]
            tips = [8, 12, 16, 20]
            knuckles = [5, 9, 13, 17]

            # Average distance to Knuckles (The "Fist" baseline)
            base_dist = sum([math.sqrt((lms[i].x - wrist.x)**2 + (lms[i].y - wrist.y)**2) for i in knuckles]) / 4
            # Average distance to Tips
            tip_dist = sum([math.sqrt((lms[i].x - wrist.x)**2 + (lms[i].y - wrist.y)**2) for i in tips]) / 4

            # 2. Throttle: Start at base_dist, Max at base_dist * 1.8 (fully extended)
            # This ensures that a fist (tips near knuckles) is 0 speed.
            speed = np.interp(tip_dist, [base_dist, base_dist * 1.8], [0.0, self.max_speed])

            # 3. Steering
            angle_rad = math.atan2(lms[12].x - wrist.x, -(lms[12].y - wrist.y))
            angle_deg = math.degrees(angle_rad)

            # 4. Apply and Publish
            if speed > 0.01:
                twist.linear.x = float(speed)
                if abs(angle_deg) > self.deadzone_deg:
                    twist.angular.z = -angle_rad * (self.sensitivity * 5.0)
            
            # Visuals
            percent = int((twist.linear.x / self.max_speed) * 100)
            cv2.putText(frame, f"Throttle: {percent}%", (10, 50), 1, 2, (0, 255, 0), 2)
            cv2.putText(frame, f"Steer: {angle_deg:.1f} deg", (10, 90), 1, 2, (255, 255, 0), 2)
            
            # Draw baseline (knuckles) for reference
            for k in knuckles:
                cv2.circle(frame, (int(lms[k].x*w), int(lms[k].y*h)), 5, (255, 0, 255), -1)

        self.publisher_.publish(twist)
        cv2.imshow("Adaptive Hand Control", frame)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = HandControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.cap.release()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()