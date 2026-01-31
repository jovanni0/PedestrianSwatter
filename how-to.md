

# launch Gazebo with world and turtlebot
```bash
export TURTLEBOT3_MODEL=waffle_pi
ros2 launch src/brainrot/launch/launch.py
```


# do this before lunching any node
cd ~/Desktop/psd_v1
source ~/venvs/ros2_vision/bin/activate
source /opt/ros/humble/setup.bash
source ~/Desktop/psd_v1/install/setup.sh


# launch the nodes
```bash
python3 src/brainrot/brainrot/arbiter.py

python3 src/brainrot/brainrot/line_folower.py

python3 src/brainrot/brainrot/stop_line.py

python3 src/brainrot/brainrot/hand_control.py
```