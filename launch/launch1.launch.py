from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_package',
            executable='controller_node.py',
            name='controller'
        ),
        Node(
            package='my_package',
            executable='guidance_node.py',
            name='guidance'
        )
    ])