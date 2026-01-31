import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import time



STOP_SIGN_CLASS = 11
STOP_DURATION = 3.0
COOLDOWN_TIME = 5.0
READY_TIMEOUT = 10.0
LINE_THRESHOLD = 1200
RED_PIXEL_THRESHOLD = 300



class VisionProcessor:

    def __init__(self):
        self.model = YOLO("yolo11n.pt")
        self.white_range = (np.array([0, 0, 180]), np.array([180, 50, 255]))
        self.r_low = (np.array([0, 50, 50]), np.array([10, 255, 255]))
        self.r_high = (np.array([160, 50, 50]), np.array([180, 255, 255]))


    def detectStopSign(self, frame, hsv):
        """
            detects stop signs in images.
        """
        results = self.model(frame, conf=0.4, verbose=False)
        h, w = frame.shape[:2]
        
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == STOP_SIGN_CLASS:
                    b = box.xyxy[0].cpu().numpy().astype(int)
                    roi = hsv[max(0, b[1]):min(h, b[3]), max(0, b[0]):min(w, b[2])]
                    mask = cv2.inRange(roi, *self.r_low) + cv2.inRange(roi, *self.r_high)
                    
                    is_red = cv2.countNonZero(mask) > RED_PIXEL_THRESHOLD

                    return True, is_red, b
                
        return False, False, None



class AutomaticStopNode(Node):

    def __init__(self):
        super().__init__("automatic_stop")

        self.sub = self.create_subscription(Image, '/camera/image_raw', self.imageCallback, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel_stop', 10)

        self.bridge = CvBridge()
        self.vision = VisionProcessor()
        
        self.sign_detected = False
        self.is_stopping = False
        self.stop_start_time = 0
        self.ready_expiry = 0
        self.last_stop_time = 0


    def imageCallback(self, data):
        """
            handle image processing for stop sign and line.
        """
        try:
            frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except: 
            return
        
        h, w, _ = frame.shape
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        now = time.time()
        in_cooldown = (now - self.last_stop_time) < COOLDOWN_TIME

        # sign detection
        if not self.is_stopping and not in_cooldown:
            found, is_red, box = self.vision.detectStopSign(frame, hsv)
            if found:
                color = (0, 0, 255) if is_red else (0, 255, 255)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 3)
                if is_red:
                    self.sign_detected = True
                    self.ready_expiry = now + READY_TIMEOUT

        # line detection
        ry1, ry2, rx1, rx2 = int(h*0.8), int(h*0.95), int(w*0.3), int(w*0.7)
        line_roi = hsv[ry1:ry2, rx1:rx2]
        line_mask = cv2.inRange(line_roi, *self.vision.white_range)
        line_detected = cv2.countNonZero(line_mask) > LINE_THRESHOLD
        
        if line_detected:
            frame[ry1:ry2, rx1:rx2][line_mask > 0] = (0, 255, 0)

        if now > self.ready_expiry: 
            self.sign_detected = False

        if self.sign_detected and line_detected and not self.is_stopping:
            self.is_stopping, self.stop_start_time, self.sign_detected = True, now, False

        self.handleStopping(frame, now, w, h)
        self.drawHud(frame, in_cooldown)

        cv2.imshow("Robot HUD", frame)
        cv2.waitKey(1)


    def handleStopping(self, frame, now, w, h):
        """
            handle the stop event.
        """
        if self.is_stopping:
            if now - self.stop_start_time < STOP_DURATION:
                self.pub.publish(Twist())
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 15)
            else:
                self.is_stopping = False
                self.last_stop_time = now


    def drawHud(self, frame, in_cooldown):
        """
            handle the HUD updates.
        """
        if self.is_stopping: status, color = "STOPPING", (0, 0, 255)
        elif in_cooldown: status, color = "COOLDOWN", (255, 0, 0)
        elif self.sign_detected: status, color = "ARMED", (0, 255, 255)
        else: status, color = "SEARCHING", (255, 255, 255)
        
        cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)



if __name__ == '__main__':
    rclpy.init()
    node = AutomaticStopNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()