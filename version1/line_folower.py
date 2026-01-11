import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class LineFolowerNode(Node):
    def __init__(self):
        super().__init__('LaneKeeperNode')
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.listener_callback, 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()

    def listener_callback(self, data):
        frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        h, w, _ = frame.shape
        
        # Honey-yellow mask (HSV)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Look only at the bottom 1/3 of the image
        mask[0:int(2*h/3), 0:w] = 0

        M = cv2.moments(mask)
        msg = Twist()

        if M['m00'] > 0:
            cx = int(M['m10']/M['m00'])
            err = cx - w/2
            # P-Control: Adjust constant and speed as needed
            msg.linear.x = 0.08 
            msg.angular.z = -float(err) / 150.0
            cv2.circle(frame, (cx, int(5*h/6)), 10, (0, 255, 0), -1)

        self.publisher.publish(msg)
        cv2.imshow("LaneKeeper View", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = LineFolowerNode()
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