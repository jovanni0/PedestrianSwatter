import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import random # For direction selection

class LineFollowerNode(Node):
    def __init__(self):
        super().__init__('LaneKeeperNode')
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.listener_callback, 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()
        self.target_index = 0 # Sticky choice for the branch

    def listener_callback(self, data):
        frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        h, w, _ = frame.shape
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Purple-Blue Mask
        lower_purple = np.array([110, 40, 40])
        upper_purple = np.array([155, 255, 255])
        mask = cv2.inRange(hsv, lower_purple, upper_purple)

        # Bottom ROI
        roi_mask = np.zeros_like(mask)
        roi_mask[int(2*h/3):h, 0:w] = 255
        search_mask = cv2.bitwise_and(mask, roi_mask)

        # Find individual branches
        contours, _ = cv2.findContours(search_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter noise
        valid_contours = [c for c in contours if cv2.contourArea(c) > 100]

        msg = Twist()

        if len(valid_contours) > 0:
            # If multiple branches found and we haven't picked one yet
            if len(valid_contours) > 1:
                self.target_index = random.randint(0, len(valid_contours) - 1)
            else:
                self.target_index = 0

            # Follow the selected branch
            M = cv2.moments(valid_contours[self.target_index])
            if M['m00'] > 0:
                cx = int(M['m10']/M['m00'])
                err = cx - w/2
                msg.linear.x = 0.3
                msg.angular.z = -float(err) / 70.0
                
                # HUD: Highlight all branches green, target branch red
                cv2.drawContours(frame, valid_contours, -1, (0, 255, 0), 2)
                cv2.circle(frame, (cx, int(5*h/6)), 7, (0, 0, 255), -1)
        else:
            msg.angular.z = 0.2 # Search

        self.publisher.publish(msg)
        cv2.imshow("Branch Selector HUD", frame)
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