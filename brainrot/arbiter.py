import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time


class CommandArbiterNode(Node):
    def __init__(self):
        super().__init__("command_arbiter")

        self.createLinks()
        
        self.last_manual_time = 0
        self.last_stop_time = 0
        
        self.manual_timeout = 0.5
        self.stop_timeout = 0.2

        print("[INFO] Command Arbiter node up and running. Awayting interrupt (Ctrl + C)...")


    def createLinks(self):
        """
        publish and subscribe the necessary topics.
        """

        self.cmd_publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        
        self.create_subscription(Twist, '/cmd_vel_auto', self.autoDrivingCallback, 10)
        self.create_subscription(Twist, '/cmd_vel_manual', self.handControlCallback, 10)
        self.create_subscription(Twist, '/cmd_vel_stop', self.signStopCallback, 10)
        

    def signStopCallback(self, msg):
        """
            handles messages published by the stop detection node.

            this callback has the highest priority and will stop the bot no matter if it drives automatically or is controlled.
        """

        self.last_stop_time = time.time()
        self.cmd_publisher.publish(msg)


    def handControlCallback(self, msg):
        """
            handles messages published by the hand control node.

            has priority over the auto driving node, but is overtaken by the stop detection node.
        """

        current_time = time.time()

        # priority check: make sure the stop node is quiet
        if (current_time - self.last_stop_time) <= self.stop_timeout:
            return
        
        if abs(msg.linear.x) > 0.01 or abs(msg.angular.z) > 0.01:
            self.last_manual_time = current_time 
            self.cmd_publisher.publish(msg)


    def autoDrivingCallback(self, msg):
        """
            handles messages published by the auto driving node.

            has the lowest priority.
        """
        now = time.time()
        if (now - self.last_stop_time > self.stop_timeout) and \
           (now - self.last_manual_time > self.manual_timeout):
            self.cmd_publisher.publish(msg)



if __name__ == '__main__':
    rclpy.init()
    rclpy.spin(CommandArbiterNode())
    rclpy.shutdown()