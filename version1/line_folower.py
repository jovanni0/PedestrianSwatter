import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import random
import time

# --- CONFIGURATION VARIABLES ---
LINEAR_SPEED = 0.4          # Increased speed
ANGULAR_GAIN = 70.0         # Steering sensitivity
COOLDOWN_TIME = 2.0         # Seconds to stick to a branch choice
MIN_AREA = 150              # Filter small noise
LOWER_PURPLE = [110, 40, 40]
UPPER_PURPLE = [155, 255, 255]
# -------------------------------

class LineFollowerNode(Node):
    def __init__(self):
        super().__init__('LaneKeeperNode')
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.listener_callback, 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()
        
        # Branch Memory
        self.last_choice_time = 0
        self.target_index = 0

    def listener_callback(self, data):
        frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        h, w, _ = frame.shape
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, np.array(LOWER_PURPLE), np.array(UPPER_PURPLE))
        
        # ROI for calculation
        roi_mask = np.zeros_like(mask)
        roi_mask[int(2*h/3):h, 0:w] = 255
        search_mask = cv2.bitwise_and(mask, roi_mask)

        contours, _ = cv2.findContours(search_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > MIN_AREA]

        msg = Twist()

        if valid_contours:
            current_time = time.time()
            
            # Only pick a new random branch if cooldown has expired
            if len(valid_contours) > 1:
                if (current_time - self.last_choice_time) > COOLDOWN_TIME:
                    self.target_index = random.randint(0, len(valid_contours) - 1)
                    self.last_choice_time = current_time
            else:
                self.target_index = 0

            # Ensure index is still valid (if a branch disappears)
            idx = min(self.target_index, len(valid_contours) - 1)
            M = cv2.moments(valid_contours[idx])
            
            if M['m00'] > 0:
                cx = int(M['m10']/M['m00'])
                err = cx - w/2
                msg.linear.x = LINEAR_SPEED
                msg.angular.z = -float(err) / ANGULAR_GAIN
                
                # HUD
                cv2.drawContours(frame, valid_contours, -1, (0, 255, 0), 2)
                cv2.circle(frame, (cx, int(5*h/6)), 7, (0, 0, 255), -1)
        else:
            msg.angular.z = 0.3 # Search spin

        self.publisher.publish(msg)
        cv2.imshow("LaneKeeper HUD", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = LineFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()