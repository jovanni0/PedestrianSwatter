import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource



def generate_launch_description():
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_tb3_gazebo = get_package_share_directory('turtlebot3_gazebo')

    launch_dir = os.path.dirname(os.path.realpath(__file__))
    pkg_models_path = os.path.join(launch_dir, '..', 'models')
    world = os.path.join(launch_dir, '..', 'worlds', 'track-v2.world')

    set_gazebo_model_path = SetEnvironmentVariable(
        name='GAZEBO_MODEL_PATH',
        value=[os.environ.get('GAZEBO_MODEL_PATH', ''), ':', pkg_models_path]
    )
    set_turtlebot_model = SetEnvironmentVariable(
        name='TURTLEBOT3_MODEL',
        value="waffle_pi"
    )

    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')),
        launch_arguments={'world': world}.items()
    )

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py'))
    )

    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_tb3_gazebo, 'launch', 'robot_state_publisher.launch.py')),
        launch_arguments={'use_sim_time': 'true'}.items()
    )

    spawn_turtlebot_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_tb3_gazebo, 'launch', 'spawn_turtlebot3.launch.py'))
    )

    ld = LaunchDescription()
    ld.add_action(set_gazebo_model_path)
    ld.add_action(set_turtlebot_model)
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(spawn_turtlebot_cmd)

    return ld