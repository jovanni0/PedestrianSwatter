import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import time

class IntegratedStopNode(Node):
    def __init__(self):
        super().__init__('integrated_stop_node')
        self.sub = self.create_subscription(Image, '/camera/image_raw', self.callback, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel_stop', 10)
        self.bridge = CvBridge()
        
        self.model = YOLO('yolo11n.pt') 
        self.LOWER_WHITE = np.array([0, 0, 200])
        self.UPPER_WHITE = np.array([180, 50, 255])
        
        self.sign_detected = False
        self.is_stopping = False
        self.stop_start_time = 0
        self.ready_timeout = 0

    def callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except: return
        
        h, w, _ = frame.shape
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        curr_time = time.time()

        # 1. Sign Detection
        sign_in_view = False
        results = self.model(frame, conf=0.5, verbose=False)
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 11: 
                    sign_in_view = True
                    self.sign_detected = True
                    self.ready_timeout = curr_time + 10.0
                        
        # 2. Line Detection
        roi_y1, roi_y2 = int(h*0.8), int(h*0.95)
        roi_x1, roi_x2 = int(w*0.3), int(w*0.7)
        roi = hsv[roi_y1:roi_y2, roi_x1:roi_x2]
        mask = cv2.inRange(roi, self.LOWER_WHITE, self.UPPER_WHITE)
        white_pixels = cv2.countNonZero(mask)
        line_detected = white_pixels > 1500

        # 3. Logic & HUD Colors
        if curr_time > self.ready_timeout: self.sign_detected = False
        
        armed = self.sign_detected
        active_stop = self.is_stopping

        if armed and line_detected and not active_stop:
            self.is_stopping = True
            self.stop_start_time = curr_time
            self.sign_detected = False 

        # 4. Publication
        if active_stop:
            if curr_time - self.stop_start_time < 3.0:
                self.pub.publish(Twist())
            else:
                self.is_stopping = False

        # --- HUD RENDERING ---
        # Draw ROI Box
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
        
        # Status Indicators
        def draw_status(label, active, pos, color_on=(0, 255, 0)):
            color = color_on if active else (50, 50, 50)
            cv2.putText(frame, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        draw_status("SIGN IN VIEW", sign_in_view, (20, 30))
        draw_status("ARMED (WAITING FOR LINE)", armed, (20, 60), (0, 255, 255))
        draw_status("LINE DETECTED", line_detected, (20, 90))
        
        if active_stop:
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)
            cv2.putText(frame, "STOPPING", (int(w/2)-80, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        cv2.imshow("Robot HUD", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = IntegratedStopNode()
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