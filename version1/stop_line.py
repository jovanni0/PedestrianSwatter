import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

class StopLineNode(Node):
    def __init__(self):
        super().__init__('stop_line_node')
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.listener_callback, 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel_stop', 10)
        self.bridge = CvBridge()

        # Config
        self.LOWER_WHITE = np.array([0, 0, 200])
        self.UPPER_WHITE = np.array([180, 50, 255])
        self.THRESHOLD = 1500  # Number of white pixels to trigger stop
        self.STOP_DURATION = 2.0
        self.COOLDOWN = 4.0

        self.is_stopping = False
        self.stop_start_time = 0
        self.last_stop_finish_time = 0

    def listener_callback(self, data):
        frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        h, w, _ = frame.shape
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        curr_time = time.time()

        # ROI: Bottom center where stop lines appear
        roi = hsv[int(h*0.75):int(h*0.9), int(w*0.2):int(w*0.8)]
        mask = cv2.inRange(roi, self.LOWER_WHITE, self.UPPER_WHITE)
        white_count = cv2.countNonZero(mask)

        msg = Twist()

        if not self.is_stopping and (curr_time - self.last_stop_finish_time > self.COOLDOWN):
            if white_count > self.THRESHOLD:
                self.is_stopping = True
                self.stop_start_time = curr_time
                self.get_logger().info("!!! STOP LINE DETECTED !!!")

        if self.is_stopping:
            if curr_time - self.stop_start_time < self.STOP_DURATION:
                # Force zero velocity
                self.publisher.publish(msg) 
            else:
                self.is_stopping = False
                self.last_stop_finish_time = curr_time

        # Debug Visuals
        cv2.rectangle(frame, (int(w*0.2), int(h*0.75)), (int(w*0.8), int(h*0.9)), (255, 0, 0), 2)
        cv2.putText(frame, f"White px: {white_count}", (20, h-20), 1, 1, (255, 255, 255), 1)
        cv2.imshow("Stop Line Detector", frame)
        cv2.waitKey(1)

def main():
    rclpy.init()
    rclpy.spin(StopLineNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()