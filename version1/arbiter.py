import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class CmdVelArbiter(Node):
    def __init__(self):
        super().__init__('cmd_vel_arbiter')
        
        # Subscribers
        self.auto_sub = self.create_subscription(Twist, '/cmd_vel_auto', self.auto_cb, 10)
        self.manual_sub = self.create_subscription(Twist, '/cmd_vel_manual', self.manual_cb, 10)
        
        # Publisher to the real robot
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # State
        self.last_manual_msg = Twist()
        self.last_manual_time = 0
        self.manual_timeout = 0.5 # Seconds before reverting to auto

    def manual_cb(self, msg):
        # Update manual state if there is actual input (moving)
        if abs(msg.linear.x) > 0.01 or abs(msg.angular.z) > 0.01:
            self.last_manual_time = time.time()
        self.last_manual_msg = msg
        self.decide_and_publish()

    def auto_cb(self, msg):
        # Only publish auto if manual hasn't been used recently
        if (time.time() - self.last_manual_time) > self.manual_timeout:
            self.cmd_pub.publish(msg)

    def decide_and_publish(self):
        # If manual is fresh, it takes over immediately
        if (time.time() - self.last_manual_time) <= self.manual_timeout:
            self.cmd_pub.publish(self.last_manual_msg)

def main():
    rclpy.init()
    node = CmdVelArbiter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()