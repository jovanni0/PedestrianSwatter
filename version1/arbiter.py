import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class CmdVelArbiter(Node):
    def __init__(self):
        super().__init__('cmd_vel_arbiter')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscriptions
        self.create_subscription(Twist, '/cmd_vel_auto', self.auto_cb, 10)
        self.create_subscription(Twist, '/cmd_vel_manual', self.manual_cb, 10)
        self.create_subscription(Twist, '/cmd_vel_stop', self.stop_cb, 10)
        
        # Timestamps for activity tracking
        self.last_manual_time = 0
        self.last_stop_time = 0
        
        self.manual_timeout = 0.5
        self.stop_timeout = 0.2

    def stop_cb(self, msg):
        """Highest Priority: Always publish stop if received."""
        self.last_stop_time = time.time()
        self.cmd_pub.publish(msg)

    def manual_cb(self, msg):
        """Medium Priority: Publish only if no Stop signal is active."""
        now = time.time()
        if abs(msg.linear.x) > 0.01 or abs(msg.angular.z) > 0.01:
            self.last_manual_time = now
            
            # Check priority: Is the stop node quiet?
            if (now - self.last_stop_time) > self.stop_timeout:
                self.cmd_pub.publish(msg)

    def auto_cb(self, msg):
        """Lowest Priority: Publish only if Stop and Manual are both quiet."""
        now = time.time()
        if (now - self.last_stop_time > self.stop_timeout) and \
           (now - self.last_manual_time > self.manual_timeout):
            self.cmd_pub.publish(msg)

def main():
    rclpy.init()
    rclpy.spin(CmdVelArbiter())
    rclpy.shutdown()

if __name__ == '__main__': main()