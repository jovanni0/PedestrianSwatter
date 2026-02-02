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
LINE_THRESHOLD = 500
RED_PIXEL_THRESHOLD = 300



class VisionProcessor:

    def __init__(self):
        self.model = YOLO("yolo11n.pt")

        self.white_range = (np.array([0, 0, 140]), np.array([180, 80, 255]))
        self.r_low = (np.array([0, 50, 50]), np.array([10, 255, 255]))
        self.r_high = (np.array([160, 50, 50]), np.array([180, 255, 255]))


    def detectStopSign(self, frame, hsv):
        """
            detects the sopt sign(s) in the frame
        """
        results = self.model(frame, conf=0.3, verbose=False)
        h, w = frame.shape[:2]

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == STOP_SIGN_CLASS:
                    b = box.xyxy[0].cpu().numpy().astype(int)
                    roi = hsv[max(0, b[1]):min(h, b[3]), max(0, b[0]):min(w, b[2])]
                    mask = cv2.inRange(roi, *self.r_low) + cv2.inRange(roi, *self.r_high)
                    return True, cv2.countNonZero(mask) > RED_PIXEL_THRESHOLD, b
                
        return False, False, None



class AutomaticStopNode(Node):

    def __init__(self):
        super().__init__("automatic_stop")

        self.createLinks()
        self.bridge = CvBridge()
        self.vision = VisionProcessor()
        self.sign_detected, self.is_stopping = False, False
        self.stop_start_time, self.ready_expiry, self.last_stop_time = 0, 0, 0


    def createLinks(self):
        self.sub = self.create_subscription(Image, '/camera/image_raw', self.imageCallback, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel_stop', 10)


    def imageCallback(self, data):
        frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        h, w, _ = frame.shape
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        now = time.time()
        in_cooldown = (now - self.last_stop_time) < COOLDOWN_TIME

        # sign detection
        if not self.is_stopping and not in_cooldown:
            found, is_red, box = self.vision.detectStopSign(frame, hsv)
            if found and is_red:
                self.sign_detected, self.ready_expiry = True, now + READY_TIMEOUT
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)

        # line detection
        ry1, ry2, rx1, rx2 = int(h*0.75), int(h*0.95), int(w*0.1), int(w*0.9)
        line_mask = cv2.inRange(hsv[ry1:ry2, rx1:rx2], *self.vision.white_range)
        px_count = cv2.countNonZero(line_mask)
        
        self.drawOverlays(frame, line_mask, px_count, rx1, ry1, rx2, ry2)

        if now > self.ready_expiry: self.sign_detected = False
        if self.sign_detected and px_count > LINE_THRESHOLD and not self.is_stopping:
            self.is_stopping, self.stop_start_time, self.sign_detected = True, now, False

        self.handleStopping(frame, now, w, h)
        self.drawHud(frame, in_cooldown)
        cv2.imshow("Detection HUD", frame)
        cv2.waitKey(1)


    def drawOverlays(self, frame, mask, px, x1, y1, x2, y2):
        """
            draw the shapes over the signs detected.
        """
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        frame[y1:y2, x1:x2][mask > 0] = (0, 255, 0)

        percent = min(1.0, px / LINE_THRESHOLD)
        cv2.rectangle(frame, (20, 60), (220, 80), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, 60), (20 + int(200 * percent), 80), (255, 255, 0), -1)


    def handleStopping(self, frame, now, w, h):
        """
            stop the bot if the sign and line are detected.
        """
        if self.is_stopping:
            if now - self.stop_start_time < STOP_DURATION:
                self.pub.publish(Twist())
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 15)
            else:
                self.is_stopping, self.last_stop_time = False, now


    def drawHud(self, frame, in_cooldown):
        """
            draw the HUD.
        """
        status = "STOPPING" if self.is_stopping else "COOLDOWN" if in_cooldown else "ARMED" if self.sign_detected else "SEARCHING"
        cv2.putText(frame, status, (20, 40), 1, 1.5, (255, 255, 255), 2)



if __name__ == '__main__':
    rclpy.init()
    rclpy.spin(AutomaticStopNode())
    rclpy.shutdown()