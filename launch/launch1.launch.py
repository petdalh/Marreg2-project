#!/usr/bin/env python3
import launch
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_package',
            executable='observer_node.py',        
            name='observer_node',
            output='screen'
        ),
        Node(
            package='my_package',
            executable='thrust_allocation_node.py',   
            name='thrust_allocation_node',
            output='screen'
        ),
        Node(
            package='my_package',
            executable='controller_node.py',        
            name='controller_node',
            output='screen',
            parameters=[{'task': 'TASK_PD_FF_CONTROLLER'}]
        ),
        Node(
            package='my_package',
            executable='guidance_node.py',        
            name='guidance_node',
            output='screen',
            parameters=[{'task': 'TASK_STATIONKEEPING'}]
        )
        Node(
            package='my_package',
            executable='joystick_node.py',        
            name='joystick_node',
            output='screen'
        )
    ])
