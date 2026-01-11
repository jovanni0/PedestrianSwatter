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
        self.WHITE_RANGE = (np.array([0, 0, 180]), np.array([180, 50, 255]))
        
        # Broad Red Range
        self.R_LOW1, self.R_HIGH1 = np.array([0, 50, 50]), np.array([10, 255, 255])
        self.R_LOW2, self.R_HIGH2 = np.array([160, 50, 50]), np.array([180, 255, 255])
        
        self.sign_detected = False
        self.is_stopping = False
        self.stop_start_time = 0
        self.ready_timeout = 0
        self.last_stop_time = 0
        self.COOLDOWN_DURATION = 5.0 # Seconds to wait after stopping

    def callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except: return
        
        h, w, _ = frame.shape
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        curr_time = time.time()
        in_cooldown = (curr_time - self.last_stop_time) < self.COOLDOWN_DURATION

        # 1. Sign Detection (Ignored if stopping or in cooldown)
        if not self.is_stopping and not in_cooldown:
            results = self.model(frame, conf=0.4, verbose=False)
            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == 11:
                        b = box.xyxy[0].cpu().numpy().astype(int)
                        roi = hsv[max(0, b[1]):min(h, b[3]), max(0, b[0]):min(w, b[2])]
                        red_mask = cv2.inRange(roi, self.R_LOW1, self.R_HIGH1) + cv2.inRange(roi, self.R_LOW2, self.R_HIGH2)
                        
                        if cv2.countNonZero(red_mask) > 300:
                            self.sign_detected = True
                            self.ready_timeout = curr_time + 10.0
                            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 3)
                        else:
                            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 3)

        # 2. Line Detection
        ry1, ry2, rx1, rx2 = int(h*0.8), int(h*0.95), int(w*0.3), int(w*0.7)
        mask = cv2.inRange(hsv[ry1:ry2, rx1:rx2], self.WHITE_RANGE[0], self.WHITE_RANGE[1])
        line_detected = cv2.countNonZero(mask) > 1200
        if line_detected:
            frame[ry1:ry2, rx1:rx2][mask > 0] = (0, 255, 0)

        # 3. HUD & Logic
        if curr_time > self.ready_timeout: self.sign_detected = False

        if self.is_stopping:
            status, color = "STOPPING", (0, 0, 255)
        elif in_cooldown:
            status, color = "COOLDOWN", (255, 0, 0)
        elif self.sign_detected:
            status, color = "ARMED", (0, 255, 255)
        else:
            status, color = "SEARCHING", (255, 255, 255)

        cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 4. Logic & Execution
        if self.sign_detected and line_detected and not self.is_stopping:
            self.is_stopping, self.stop_start_time, self.sign_detected = True, curr_time, False

        if self.is_stopping:
            if curr_time - self.stop_start_time < 3.0:
                self.pub.publish(Twist())
                cv2.rectangle(frame, (0,0), (w,h), (0,0,255), 15)
            else:
                self.is_stopping = False
                self.last_stop_time = curr_time

        cv2.imshow("Robot HUD", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = IntegratedStopNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()