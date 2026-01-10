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

        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)
        self.mp_draw = mp.solutions.drawing_utils

        # Configuration
        self.sensitivity = 0.6 
        self.linear_speed = 0.2
        self.deadzone_deg = 5.0 # Degrees to ignore for straight driving

    def run_loop(self):
        success, frame = self.cap.read()
        if not success: return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        result = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        twist = Twist()

        if result.multi_hand_landmarks:
            lms = result.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(frame, lms, self.mp_hands.HAND_CONNECTIONS)

            # Finger Count Logic
            fingers = sum(1 for t, p in zip([8,12,16,20], [6,10,14,18]) 
                         if lms.landmark[t].y < lms.landmark[p].y)
            
            # Gesture and Movement Logic
            if fingers >= 3:
                gesture_text = "MOVE (OPEN)"
                color = (0, 255, 0) # Green
                
                # Calculate Steering only if hand is open
                wrist, tip = lms.landmark[0], lms.landmark[12]
                angle_rad = math.atan2(tip.x - wrist.x, -(tip.y - wrist.y))
                angle_deg = math.degrees(angle_rad)
                
                # Apply Deadzone
                if abs(angle_deg) > self.deadzone_deg:
                    twist.angular.z = -angle_rad * (self.sensitivity * 5.0)
                
                twist.linear.x = self.linear_speed
                
                # Draw steering line
                p1 = (int(wrist.x * w), int(wrist.y * h))
                p2 = (int(tip.x * w), int(tip.y * h))
                cv2.line(frame, p1, p2, (255, 0, 0), 3)
                cv2.putText(frame, f"Steer: {angle_deg:.1f} deg", (10, 90), 1, 2, (255, 255, 0), 2)
            else:
                gesture_text = "STOP (FIST)"
                color = (0, 0, 255) # Red
                # Twist values remain 0.0
            
            cv2.putText(frame, f"State: {gesture_text}", (10, 50), 1, 2, color, 2)

        self.publisher_.publish(twist)
        cv2.imshow("TurtleBot Hand Control", frame)
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