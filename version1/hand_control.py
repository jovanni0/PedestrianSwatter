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
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel_manual', 10)
        self.timer = self.create_timer(0.033, self.run_loop)

        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.8, max_num_hands=1)
        self.mp_draw = mp.solutions.drawing_utils

        # Config
        self.sensitivity = 0.2 
        self.max_speed = 0.8
        self.deadzone_deg = 5.0
        self.hand_limit_deg = 30.0 # Steering caps at this hand tilt
        self.forward_percentage = 0.8 # how big to be the reverse zone
        
        # Play/Pause Logic
        self.is_paused = True
        self.last_toggle_time = 0
        self.cooldown_duration = 2.0 

        # Smoothing & State
        self.prev_linear = 0.0
        self.prev_angular = 0.0
        self.alpha = 0.2  # Smoothing factor (0.1 = very smooth, 1.0 = no smoothing)

    def run_loop(self):
        success, frame = self.cap.read()
        if not success: return
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        result = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        twist = Twist()

        # Permanent HUD: Black Top Bar
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)

        # Suggestion 4: Visual separator for Reverse (at 70% height)
        cv2.line(frame, (0, int(h * self.forward_percentage)), (w, int(h * self.forward_percentage)), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "REVERSE ZONE", (10, int(h * self.forward_percentage) + 20), 1, 1, (255, 255, 255), 1)

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
                base_dist = sum([math.sqrt((lms[i].x-wrist.x)**2 + (lms[i].y-wrist.y)**2) for i in knuckles]) / 4
                tip_dist = sum([math.sqrt((lms[i].x-wrist.x)**2 + (lms[i].y-wrist.y)**2) for i in tips]) / 4

                # Suggestion 4: Vertical Reverse (Wrist in bottom 30% of image)
                direction = -1.0 if wrist.y > self.forward_percentage else 1.0
                speed = np.interp(tip_dist, [base_dist * 1.1, base_dist * 1.8], [0.0, self.max_speed])
                
                # Capped Steering
                angle_rad = math.atan2(lms[12].x - wrist.x, -(lms[12].y - wrist.y))
                angle_deg = math.degrees(angle_rad)
                
                # Clip input to the limit and scale sensitivity for faster response
                capped_angle_deg = np.clip(angle_deg, -self.hand_limit_deg, self.hand_limit_deg)
                
                # Calculation with smoothing (Suggestion 2)
                target_linear = float(speed * direction)
                target_angular = -math.radians(capped_angle_deg) * (self.sensitivity * 8.0)
                
                twist.linear.x = (self.alpha * target_linear) + ((1 - self.alpha) * self.prev_linear)
                twist.angular.z = (self.alpha * target_angular) + ((1 - self.alpha) * self.prev_angular)

                # Active HUD elements
                speed_pct = int((abs(twist.linear.x) / self.max_speed) * 100)
                steer_pct = int((capped_angle_deg / self.hand_limit_deg) * 100)
                dir_label = "REV" if direction < 0 else "FWD"
                cv2.putText(frame, f"{dir_label} {speed_pct}%", (w - 320, 40), 1, 1.8, (255, 255, 255), 2)
                cv2.putText(frame, f"STR: {steer_pct}%", (w - 160, 40), 1, 1.8, (255, 255, 255), 2)

        # Suggestion 1: Deadman's Switch (Implicitly stops if no landmarks or paused)
        self.prev_linear = twist.linear.x
        self.prev_angular = twist.angular.z

        # Permanent Status Display
        status_text = "LOCKED" if self.is_paused else "ACTIVE"
        color = (0, 0, 255) if self.is_paused else (0, 255, 0)
        cv2.putText(frame, f"STATUS: {status_text}", (20, 40), 1, 1.8, color, 2)
        
        # Cooldown Indicator
        elapsed = time.time() - self.last_toggle_time
        if elapsed < self.cooldown_duration:
            bar_w = int((elapsed / self.cooldown_duration) * 150)
            cv2.rectangle(frame, (20, 50), (20 + bar_w, 55), (255, 255, 0), -1)

        self.publisher_.publish(twist)
        cv2.imshow("WafflePi HUD", frame)
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