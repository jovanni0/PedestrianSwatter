import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import random
import time

# --- CONFIGURATION ---
LINEAR_SPEED = 0.3
ANGULAR_GAIN = 80.0
COOLDOWN_TIME = 3.0
DECISION_DELAY = 0.50        
MIN_AREA = 180
LOWER_PURPLE = [110, 40, 40]
UPPER_PURPLE = [155, 255, 255]

TRACK_START, TRACK_END = 0.80, 1.0
LOOK_START,  LOOK_END  = 0.66, 0.79
# ---------------------

class LineFollowerFinal(Node):
    def __init__(self):
        super().__init__('LaneKeeperNode')
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.listener_callback, 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel_auto', 10)
        self.bridge = CvBridge()
        
        self.last_choice_time = 0
        self.branch_seen_start = 0
        self.target_side = "center" 

    def listener_callback(self, data):
        frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        h, w, _ = frame.shape
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(LOWER_PURPLE), np.array(UPPER_PURPLE))
        curr_time = time.time()

        # --- SELECTION ZONE ---
        look_mask = np.zeros_like(mask)
        look_mask[int(h*LOOK_START):int(h*LOOK_END), :] = 255
        look_res = cv2.bitwise_and(mask, look_mask)
        l_cnts, _ = cv2.findContours(look_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_look = [c for c in l_cnts if cv2.contourArea(c) > MIN_AREA]

        cooldown_rem = max(0, COOLDOWN_TIME - (curr_time - self.last_choice_time))

        # Branch Logic & Countdown Calculation
        decision_timer = 0
        if len(valid_look) >= 2 and cooldown_rem <= 0:
            if self.branch_seen_start == 0:
                self.branch_seen_start = curr_time
            
            decision_timer = DECISION_DELAY - (curr_time - self.branch_seen_start)
            
            if decision_timer <= 0:
                self.target_side = "left" if random.random() < 0.5 else "right"
                self.last_choice_time = curr_time
                self.branch_seen_start = 0
        else:
            self.branch_seen_start = 0

        # --- TRACKING ZONE ---
        track_mask = np.zeros_like(mask)
        track_mask[int(h*TRACK_START):int(h*TRACK_END), :] = 255
        track_res = cv2.bitwise_and(mask, track_mask)
        t_cnts, _ = cv2.findContours(track_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_track = sorted([c for c in t_cnts if cv2.contourArea(c) > MIN_AREA], 
                            key=lambda c: cv2.moments(c)['m10']/cv2.moments(c)['m00'] if cv2.moments(c)['m00'] > 0 else 0)

        msg = Twist()
        line_found = False

        if valid_track:
            # SAFETY: Ensure index exists
            idx = 0 if self.target_side == "left" or len(valid_track) == 1 else -1
            try:
                M = cv2.moments(valid_track[idx])
                if M['m00'] > 0:
                    cx = int(M['m10']/M['m00'])

                    # Validation: Don't snap to a line on the opposite side of our lock
                    is_valid = True
                    if self.target_side == "left" and cx > (w * 0.6): is_valid = False
                    if self.target_side == "right" and cx < (w * 0.4): is_valid = False

                    if is_valid:
                        msg.linear.x = LINEAR_SPEED
                        msg.angular.z = -float(cx - w/2) / ANGULAR_GAIN
                        cv2.drawContours(frame, [valid_track[idx]], -1, (0, 255, 0), 3)
                        line_found = True
            except IndexError:
                pass 

        if not line_found:
            # --- RECOVERY SPIN ---
            # If line lost, spin in the direction of our last decision
            msg.linear.x = 0.0
            msg.angular.z = 0.4 if self.target_side == "left" else -0.4

        # Reset target side once cooldown expires
        if cooldown_rem <= 0:
            self.target_side = "center"

        # --- HUD ENHANCEMENTS ---
        # Draw Selection Zone Contours in Yellow
        for c in valid_look:
            cv2.drawContours(frame, [c], -1, (0, 255, 255), 2)

        # Labels
        cv2.putText(frame, f"ACTIVE: {self.target_side.upper()}", (20, 40), 1, 1.2, (0, 255, 255), 2)
        
        # Cooldown Bar (Yellow)
        bar_len = int((cooldown_rem / COOLDOWN_TIME) * 150)
        cv2.putText(frame, "COOLDOWN", (180, 58), 1, 0.8, (255, 255, 255), 1)
        cv2.rectangle(frame, (20, 50), (170, 60), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, 50), (20 + bar_len, 60), (0, 255, 255), -1)
        
        # Decision Timer (Red)
        if decision_timer > 0:
            timer_w = int((decision_timer / DECISION_DELAY) * 150)
            cv2.putText(frame, f"DECIDING IN: {decision_timer:.1f}s", (20, 85), 1, 1.2, (0, 0, 255), 2)
            cv2.rectangle(frame, (20, 95), (170, 105), (50, 50, 50), -1)
            cv2.rectangle(frame, (20, 95), (20 + timer_w, 105), (0, 0, 255), -1)

        self.publisher.publish(msg)
        cv2.imshow("LaneKeeper HUD", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = LineFollowerFinal()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()