import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class LaneKeeperNode(Node):
    def __init__(self):
        super().__init__('camera_viewer')
        # Waffle Pi default camera topic is usually /camera/image_raw
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.listener_callback, 10)
        self.bridge = CvBridge()

    def listener_callback(self, data):
        # Convert ROS Image message to OpenCV image
        current_frame = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        cv2.imshow("TurtleBot Waffle Pi Camera", current_frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = LaneKeeperNode()
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