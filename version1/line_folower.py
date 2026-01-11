import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class LineFollowerNode(Node):
    def __init__(self):
        super().__init__('LaneKeeperNode')
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.listener_callback, 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()

    def listener_callback(self, data):
        frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        h, w, _ = frame.shape
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Broadened Purple-Blue range for the specific track color
        lower_purple = np.array([110, 40, 40])
        upper_purple = np.array([155, 255, 255])
        mask = cv2.inRange(hsv, lower_purple, upper_purple)

        # Separate ROI for calculation vs HUD display
        search_mask = mask.copy()
        search_mask[0:int(2*h/3), 0:w] = 0 

        # HUD: Draw the mask boundaries and highlight line
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2) # Green outline on line

        M = cv2.moments(search_mask)
        msg = Twist()

        if M['m00'] > 0:
            cx = int(M['m10']/M['m00'])
            err = cx - w/2
            msg.linear.x = 0.1  # Increased speed slightly
            msg.angular.z = -float(err) / 80.0 # Higher gain for tighter turns
            cv2.circle(frame, (cx, int(5*h/6)), 10, (0, 0, 255), -1) # Red tracking dot
        else:
            # Spin slowly to find the line if lost
            msg.angular.z = 0.2

        self.publisher.publish(msg)
        cv2.imshow("Track HUD", frame)
        cv2.imshow("Mask Debug", mask) # Helpful to see if the colors match
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