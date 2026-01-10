#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import cv2
import mediapipe as mp
import math
import numpy as np
import time

class HandControlNode(Node):
    def __init__(self):
        super().__init__('hand_control_node')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.033, self.run_loop)

        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.8, max_num_hands=1)
        self.mp_draw = mp.solutions.drawing_utils

        # Config
        self.sensitivity = 0.2 
        self.max_speed = 0.8
        self.deadzone_deg = 5.0
        
        # Play/Pause Logic
        self.is_paused = True
        self.last_toggle_time = 0
        self.cooldown_duration = 2.0 

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

            # 1. STRICT Pointing Gesture Detection
            # Index is ONLY considered "pointing" if it's way above the middle finger knuckle
            index_extended = lms[8].y < lms[6].y and lms[8].y < lms[5].y
            
            # Others (Middle, Ring, Pinky) must be deeply curled (tips below their own MCP/knuckle joints)
            others_curled = all(lms[t].y > lms[mcp].y for t, mcp in zip([12, 16, 20], [9, 13, 17]))
            
            # Thumb should also be tucked in to be sure
            thumb_tucked = lms[4].x > lms[13].x if lms[17].x > lms[5].x else lms[4].x < lms[13].x

            if index_extended and others_curled and thumb_tucked:
                current_time = time.time()
                if (current_time - self.last_toggle_time) > self.cooldown_duration:
                    self.is_paused = not self.is_paused
                    self.last_toggle_time = current_time
                    # Visual feedback for toggle
                    cv2.circle(frame, (int(lms[8].x*w), int(lms[8].y*h)), 20, (255, 255, 255), -1)

            # 2. Control Logic
            if not self.is_paused:
                wrist = lms[0]
                tips = [8, 12, 16, 20]
                knuckles = [5, 9, 13, 17]
                base_dist = sum([math.sqrt((lms[i].x - wrist.x)**2 + (lms[i].y - wrist.y)**2) for i in knuckles]) / 4
                tip_dist = sum([math.sqrt((lms[i].x - wrist.x)**2 + (lms[i].y - wrist.y)**2) for i in tips]) / 4

                speed = np.interp(tip_dist, [base_dist * 1.1, base_dist * 1.8], [0.0, self.max_speed])
                angle_rad = math.atan2(lms[12].x - wrist.x, -(lms[12].y - wrist.y))
                
                if speed > 0.01:
                    twist.linear.x = float(speed)
                    twist.angular.z = -angle_rad * (self.sensitivity * 2.0)

            # 3. Visuals
            status_text = "LOCKED" if self.is_paused else "ACTIVE"
            color = (0, 0, 255) if self.is_paused else (0, 255, 0)
            cv2.putText(frame, status_text, (20, 50), 1, 2, color, 3)
            
            # Cooldown progress bar
            elapsed = time.time() - self.last_toggle_time
            if elapsed < self.cooldown_duration:
                bar_w = int((elapsed / self.cooldown_duration) * 200)
                cv2.rectangle(frame, (20, 70), (20 + bar_w, 80), (255, 255, 0), -1)

        self.publisher_.publish(twist)
        cv2.imshow("Strict Hand Control", frame)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = HandControlNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    node.cap.release()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()