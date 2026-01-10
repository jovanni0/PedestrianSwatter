#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import cv2
import mediapipe as mp
import math

class HandControlNode(Node):
    def __init__(self):
        super().__init__('hand_control_node')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.033, self.run_loop)

        # MediaPipe & CV Setup
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils

        # Configuration
        self.sensitivity = 0.6  # Adjust to change steering responsiveness
        self.linear_speed = 0.2

    def run_loop(self):
        success, frame = self.cap.read()
        if not success: return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        
        twist = Twist()

        if result.multi_hand_landmarks:
            lms = result.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(frame, lms, self.mp_hands.HAND_CONNECTIONS)

            # 1. Steering Logic (Angle between Wrist 0 and Middle Tip 12)
            wrist = lms.landmark[0]
            tip = lms.landmark[12]
            angle_rad = math.atan2(tip.x - wrist.x, -(tip.y - wrist.y))
            
            # Apply sensitivity and publish
            twist.angular.z = -angle_rad * (self.sensitivity * 5.0)

            # 2. Throttle Logic (Finger count)
            # Counts tips (8,12,16,20) above their pips (6,10,14,18)
            fingers = sum(1 for t, p in zip([8,12,16,20], [6,10,14,18]) 
                         if lms.landmark[t].y < lms.landmark[p].y)
            
            gesture = "FIST"
            if fingers >= 3:
                twist.linear.x = self.linear_speed
                gesture = "OPEN HAND"

            # 3. Visualization
            deg = math.degrees(angle_rad)
            cv2.line(frame, (int(wrist.x*w), int(wrist.y*h)), (int(tip.x*w), int(tip.y*h)), (255,0,0), 3)
            cv2.putText(frame, f"Gesture: {gesture}", (10, 50), 1, 2, (0, 255, 0), 2)
            cv2.putText(frame, f"Angle: {deg:.1f} deg", (10, 90), 1, 2, (255, 255, 0), 2)

        self.publisher_.publish(twist)
        cv2.imshow("Hand Control & Visualizer", frame)
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