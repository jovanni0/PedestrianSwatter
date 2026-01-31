

# launch Gazebo with world and turtlebot
export TURTLEBOT3_MODEL=waffle_pi
ros2 launch turtlebot3_gazebo empty_world.launch.py


# do this before lunching any node
cd ~/Desktop/psd_v1
source ~/venvs/ros2_vision/bin/activate
source /opt/ros/humble/setup.bash
source ~/Desktop/psd_v1/install/setup.sh


# launch the hand control node
python3 src/version1/version1/hand_control.py

python3 src/version1/version1/line_folower.py

python3 src/version1/version1/arbiter.py

python3 src/version1/version1/stop_line.py