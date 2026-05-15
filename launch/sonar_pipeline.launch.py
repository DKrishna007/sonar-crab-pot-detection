#!/usr/bin/env python3
"""sonar_pipeline.launch.py - Launch full sonar perception pipeline."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
      return LaunchDescription([
                DeclareLaunchArgument('scan_width',      default_value='1024'),
                DeclareLaunchArgument('min_confidence',  default_value='0.35'),
                DeclareLaunchArgument('max_buffer_age_s',default_value='2.0'),
                DeclareLaunchArgument('min_area',        default_value='40.0'),
                DeclareLaunchArgument('max_area',        default_value='4000.0'),

                LogInfo(msg='Starting sonar perception pipeline...'),

                Node(
                              package='sonar_perception',
                              executable='packet_stitcher',
                              name='packet_stitcher',
                              output='screen',
                              parameters=[{
                                                'scan_width':       LaunchConfiguration('scan_width'),
                                                'max_buffer_age_s': LaunchConfiguration('max_buffer_age_s'),
                              }],
                ),
                Node(
                              package='sonar_perception',
                              executable='detector',
                              name='crab_pot_detector',
                              output='screen',
                              parameters=[{
                                                'min_confidence': LaunchConfiguration('min_confidence'),
                                                'min_area':       LaunchConfiguration('min_area'),
                                                'max_area':       LaunchConfiguration('max_area'),
                              }],
                ),
                Node(
                              package='sonar_perception',
                              executable='gnss_imu_fusion',
                              name='gnss_imu_fusion',
                              output='screen',
                              remappings=[
                                                ('/fix',      '/gps/fix'),
                                                ('/imu/data', '/imu/data'),
                                                ('/odom',     '/wheel_odometry'),
                              ],
                ),
      ])
  
