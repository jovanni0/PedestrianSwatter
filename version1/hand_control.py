import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import cv2
import mediapipe as mp
import math
import numpy as np
import time


class HandProcessor:

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.8, max_num_hands=1)
        self.mp_draw = mp.solutions.drawing_utils


    def getHandData(self, frame):
        """
            returns the hand data from the image it a hand exists.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        return result.multi_hand_landmarks



class HandControlNode(Node):

    def __init__(self):
        super().__init__("hand_control")

        # Config
        self.fps = 30  # how many frames to process in a second.
        self.max_speed = 0.4  # the max speed of the bot.
        self.hand_limit_deg = 60.0  # steering caps at this hand tilt. the smaller the limit, the more sensitive it is.
        self.forward_percentage = 0.8  # how big to be the reverse zone is on the bottom of the screen.
        self.cooldown_duration = 2.0  # cooldown duration between the moment the enable sign is detected and when the commands are being read.
        self.movement_smoothing = 0.2  # how much the movement should be smoothed (0.1 = very smooth, 1.0 = no smoothing).
        self.steering_sensitivity = 0.2  # how sensitive the bot should be at steering input change.
        self.steering_deadzone = 0.0  # a central deadzone for steering.
        self.steering_offset = -10.0  # offsets the 0 position of the hand by the given degrees.


        self.is_paused = True
        self.last_toggle_time = 0
        self.prev_linear = 0.0
        self.prev_angular = 0.0

        self.hand_processor = HandProcessor()

        self.publisher_ = self.create_publisher(Twist, "/cmd_vel_manual", 10)
        self.timer = self.create_timer(1 / self.fps, self.processLoop)

        self.openCamera()

        print("[INFO] Hand Control node running. Awayting interrupt (Ctrl + C)...")


    def openCamera(self):
        """
            captures the first camera it finds.
        """

        for index in range(10):
            self.cap = cv2.VideoCapture(index)

            if self.cap.isOpened():
                print(f"Camera found at index: {index}")
                break

        if (not self.cap.isOpened()):
            print("[ERROR]: Could not find camera to open.")


    def processLoop(self):
        """
            process each frame and send twist commands.
        """
        success, frame = self.cap.read()

        if not success: 
            print("[WARNING] Could not read frame")
            return
        
        twist = Twist()
        frame = cv2.flip(frame, 1)

        result = self.hand_processor.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        hands_landmarks = result.multi_hand_landmarks

        self.drawHud(frame)
        
        if not hands_landmarks:
            self.putMessageOnImageCenter(frame, "No hands detected")
        else:
            self.processHandSignal(frame, hands_landmarks, twist)

        self.prev_linear = twist.linear.x
        self.prev_angular = twist.angular.z

        self.publisher_.publish(twist)

        cv2.imshow("WafflePi HUD", frame)
        cv2.waitKey(1)


    def processHandSignal(self, frame, hands_landmarks, twist):
        """
            processed the hand position and sends command signals.
        """
        h, w, _ = frame.shape
        hand_landmarks = hands_landmarks[0].landmark

        self.drawHandLandmarks(frame, hands_landmarks)

        current_time = time.time()
        if self.isToggleGesture(hand_landmarks) and (current_time - self.last_toggle_time) > self.cooldown_duration:
            self.is_paused = not self.is_paused
            self.last_toggle_time = current_time

            cv2.circle(frame, (int(hand_landmarks[8].x*w), int(hand_landmarks[8].y*h)), 20, (255, 255, 255), -1)

        if self.is_paused:
            return
        
        x_twist, z_twist, capped_angle, direction = self.calculateTwist(hand_landmarks)
        twist.linear.x = x_twist
        twist.angular.z = z_twist


        speed_pct = int((abs(x_twist) / self.max_speed) * 100)
        steer_pct = int((capped_angle / self.hand_limit_deg) * 100)
        dir_label = "REV" if direction < 0 else "FWD"

        self.drawActiveHud(frame, speed_pct, steer_pct, dir_label)


    def drawHandLandmarks(self, frame, hands_landmarks):
        hand_landmarks = hands_landmarks[0].landmark

        self.hand_processor.mp_draw.draw_landmarks(
            frame, hands_landmarks[0], 
            self.hand_processor.mp_hands.HAND_CONNECTIONS
        )

        # draw the 0 steering line
        h, w = frame.shape[:2]
        wrist_pix = (int(hand_landmarks[0].x * w), int(hand_landmarks[0].y * h))
        index_mcp_pix = (int(hand_landmarks[8].x * w), int(hand_landmarks[8].y * h))

        length = math.sqrt((wrist_pix[0] - index_mcp_pix[0])**2 + (wrist_pix[1] - index_mcp_pix[1])**2)

        angle_rad = math.radians(self.steering_offset - 90)
        end_x = int(wrist_pix[0] + length * math.cos(angle_rad))
        end_y = int(wrist_pix[1] + length * math.sin(angle_rad))

        cv2.line(frame, wrist_pix, (end_x, end_y), (0, 255, 255), 3) # Yellow line
        cv2.putText(frame, "ZERO", (end_x, end_y - 10), 1, 1, (0, 255, 255), 1)



    def putMessageOnImageCenter(self, frame, text, font = cv2.FONT_HERSHEY_SIMPLEX, scale = 1.0, color = (255, 255, 255), thickness = 2):
        """
            writed a message on the center of the screen
        """
        h, w = frame.shape[:2]
        
        (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
        
        x = (w - text_w) // 2
        y = (h + text_h) // 2
        
        cv2.putText(frame, text, (x, y), font, scale, color, thickness)


    def drawHud(self, frame):
        """
            draws the basic HUD elements.
        """
        h, w = frame.shape[:2]

        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.line(frame, (0, int(h * self.forward_percentage)), (w, int(h * self.forward_percentage)), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "REVERSE ZONE", (10, int(h * self.forward_percentage) + 20), 1, 1, (255, 255, 255), 1)

        # control state
        if self.is_paused:
            cv2.putText(frame, "STATUS: LOCKED", (20, 40), 1, 1.8, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "STATUS: ACTIVE", (20, 40), 1, 1.8, (0, 255, 0), 2)

        # control state cooldown
        elapsed = time.time() - self.last_toggle_time
        if elapsed < self.cooldown_duration:
            bar_w = int((elapsed / self.cooldown_duration) * 150)
            cv2.rectangle(frame, (20, 50), (20 + bar_w, 55), (255, 255, 0), -1)


    def drawActiveHud(self, frame, speed, steer, direction):
        """
            draws the active HUD elements, like speed, steering and direction.
        """
        h, w = frame.shape[:2]

        cv2.putText(frame, f"{direction} {speed}%", (w - 320, 40), 1, 1.8, (255, 255, 255), 2)
        cv2.putText(frame, f"STR: {steer}%", (w - 160, 40), 1, 1.8, (255, 255, 255), 2)


    def isToggleGesture(self, hand_landmarks):
        """
            checks if the hand makes the state toggle gesture.
        """
        index_extended = (hand_landmarks[8].y < hand_landmarks[6].y) and (hand_landmarks[8].y < hand_landmarks[5].y)
        others_curled = all(hand_landmarks[t].y > hand_landmarks[mcp].y for t, mcp in zip([12, 16, 20], [9, 13, 17]))
        thumb_tucked = hand_landmarks[4].x > hand_landmarks[13].x if hand_landmarks[17].x > hand_landmarks[5].x else hand_landmarks[4].x < hand_landmarks[13].x

        return index_extended and others_curled and thumb_tucked
        

    def calculateTwist(self, hand_landmarks):
        """
            calculates the twist values based on the hand position.

            also determines the direction of movement and the steering angle.
        """
        wrist = hand_landmarks[0]
        tips = [8, 12, 16, 20]
        knuckles = [5, 9, 13, 17]
        base_dist = sum([math.sqrt((hand_landmarks[i].x-wrist.x)**2 + (hand_landmarks[i].y-wrist.y)**2) for i in knuckles]) / 4
        tip_dist = sum([math.sqrt((hand_landmarks[i].x-wrist.x)**2 + (hand_landmarks[i].y-wrist.y)**2) for i in tips]) / 4

        direction = -1.0 if wrist.y > self.forward_percentage else 1.0
        speed = np.interp(tip_dist, [base_dist * 1.1, base_dist * 1.8], [0.0, self.max_speed])

        # cap the steering angle and add deadzone
        angle_rad = math.atan2(hand_landmarks[12].x - wrist.x, -(hand_landmarks[12].y - wrist.y))
        angle_deg = math.degrees(angle_rad) - self.steering_offset

        if abs(angle_deg) < self.steering_deadzone:
            angle_deg = 0.0
        else:
            angle_deg = np.sign(angle_deg) * (abs(angle_deg) - self.steering_deadzone)

        capped_angle_deg = np.clip(angle_deg, -self.hand_limit_deg, self.hand_limit_deg)
        
        # smooth the movement
        target_linear = float(speed * direction)
        target_angular = -math.radians(capped_angle_deg) * (self.steering_sensitivity * 8.0)
        
        x_twist = (self.movement_smoothing * target_linear) + ((1 - self.movement_smoothing) * self.prev_linear)
        z_twist = (self.movement_smoothing * target_angular) + ((1 - self.movement_smoothing) * self.prev_angular)

        return x_twist, z_twist, capped_angle_deg, direction





if __name__ == '__main__':
    rclpy.init()
    node = HandControlNode()

    try: 
        rclpy.spin(node)
    except KeyboardInterrupt: 
        pass
    
    node.cap.release()
    cv2.destroyAllWindows()
    rclpy.shutdown()