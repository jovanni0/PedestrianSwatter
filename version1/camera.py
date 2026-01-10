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

        # Configuration
        self.sensitivity = 0.6 
        self.max_speed = 0.26
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

            # 1. Modulation Logic: Distance from Wrist (0) to Middle Finger Tip (12)
            wrist = lms[0]
            middle_tip = lms[12]
            dist = math.sqrt((middle_tip.x - wrist.x)**2 + (middle_tip.y - wrist.y)**2)
            
            # Map distance (closed ~0.15 to open ~0.45) to speed
            # These values might need slight tuning based on your distance from camera
            speed = np.interp(dist, [0.2, 0.45], [0.0, self.max_speed])
            
            # 2. Steering Logic
            angle_rad = math.atan2(middle_tip.x - wrist.x, -(middle_tip.y - wrist.y))
            angle_deg = math.degrees(angle_rad)

            # 3. Stop Condition (Fist check)
            fingers_open = sum(1 for t, p in zip([8,12,16,20], [6,10,14,18]) if lms[t].y < lms[p].y)
            
            if fingers_open > 0:
                twist.linear.x = float(speed)
                if abs(angle_deg) > self.deadzone_deg:
                    twist.angular.z = -angle_rad * (self.sensitivity * 5.0)
                state_text = f"Moving: {int((speed/self.max_speed)*100)}%"
                color = (0, 255, 0)
            else:
                state_text = "STOPPED (FIST)"
                color = (0, 0, 255)

            # Visuals
            cv2.putText(frame, state_text, (10, 50), 1, 2, color, 2)
            cv2.putText(frame, f"Steer: {angle_deg:.1f} deg", (10, 90), 1, 2, (255, 255, 0), 2)
            cv2.line(frame, (int(wrist.x*w), int(wrist.y*h)), (int(middle_tip.x*w), int(middle_tip.y*h)), (255, 0, 0), 3)

        self.publisher_.publish(twist)
        cv2.imshow("Modulated Control", frame)
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