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
LINE_THRESHOLD = 500      # Pixels needed to trigger
RED_PIXEL_THRESHOLD = 300

class VisionProcessor:
    def __init__(self):
        # Use a lightweight model
        self.model = YOLO("yolo11n.pt")
        # Loosened White: V lowered to 140 to catch greyish white lines
        self.white_range = (np.array([0, 0, 140]), np.array([180, 80, 255]))
        self.r_low = (np.array([0, 50, 50]), np.array([10, 255, 255]))
        self.r_high = (np.array([160, 50, 50]), np.array([180, 255, 255]))

    def detectStopSign(self, frame, hsv):
        results = self.model(frame, conf=0.3, verbose=False)
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
        try:
            frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except: return
        
        h, w, _ = frame.shape
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        now = time.time()
        in_cooldown = (now - self.last_stop_time) < COOLDOWN_TIME

        # 1. Sign Detection (ARMING)
        if not self.is_stopping and not in_cooldown:
            found, is_red, box = self.vision.detectStopSign(frame, hsv)
            if found and is_red:
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)
                if not self.sign_detected:
                    print("[INFO] NODE ARMED")
                self.sign_detected = True
                self.ready_expiry = now + READY_TIMEOUT

        # 2. Line Detection & Visual Highlighting
        # ROI: Bottom of the screen where the stop line usually appears
        ry1, ry2, rx1, rx2 = int(h*0.75), int(h*0.95), int(w*0.1), int(w*0.9)
        
        # Draw search area (Blue Box)
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
        
        line_roi = hsv[ry1:ry2, rx1:rx2]
        line_mask = cv2.inRange(line_roi, *self.vision.white_range)
        px_count = cv2.countNonZero(line_mask)
        line_detected = px_count > LINE_THRESHOLD

        # --- PROGRESS BAR CODE ---
        percent = min(1.0, px_count / LINE_THRESHOLD)
        bar_w = int(200 * percent)
        cv2.rectangle(frame, (20, 60), (220, 80), (50, 50, 50), -1) # Background
        bar_color = (0, 255, 0) if percent >= 1.0 else (255, 255, 0)
        cv2.rectangle(frame, (20, 60), (20 + bar_w, 80), bar_color, -1) # Progress
        cv2.putText(frame, f"{int(percent*100)}%", (230, 75), 1, 1, (255, 255, 255), 1)
        # -------------------------
        
        # Highlight white pixels found (Green Overlay)
        roi_bgr = frame[ry1:ry2, rx1:rx2]
        roi_bgr[line_mask > 0] = (0, 255, 0)
        
        # Display Pixel Count
        cv2.putText(frame, f"White Px: {px_count}", (rx1, ry1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 3. Trigger Logic
        if now > self.ready_expiry and self.sign_detected: 
            self.sign_detected = False
            print("[INFO] DISARMED (Timeout)")

        if self.sign_detected and line_detected and not self.is_stopping:
            print(f"[STOP] Line detected with {px_count} pixels!")
            self.is_stopping = True
            self.stop_start_time = now
            self.sign_detected = False

        self.handleStopping(frame, now, w, h)
        self.drawHud(frame, in_cooldown)
        
        cv2.imshow("Detection HUD", frame)
        cv2.waitKey(1)

    def handleStopping(self, frame, now, w, h):
        if self.is_stopping:
            if now - self.stop_start_time < STOP_DURATION:
                self.pub.publish(Twist())
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 15)
                cv2.putText(frame, "!!! STOP !!!", (w//2-100, h//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            else:
                self.is_stopping = False
                self.last_stop_time = now

    def drawHud(self, frame, in_cooldown):
        if self.is_stopping: status, color = "STOPPING", (0, 0, 255)
        elif in_cooldown: status, color = "COOLDOWN", (255, 0, 0)
        elif self.sign_detected: status, color = "ARMED", (0, 255, 255)
        else: status, color = "SEARCHING", (255, 255, 255)
        cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

if __name__ == '__main__':
    rclpy.init()
    node = AutomaticStopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()