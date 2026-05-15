from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'sonar_perception'

setup(
      name=package_name,
      version='1.0.0',
      packages=find_packages(exclude=['test']),
      data_files=[
                ('share/ament_index/resource_index/packages',
                             ['resource/' + package_name]),
                ('share/' + package_name, ['package.xml']),
                (os.path.join('share', package_name, 'launch'),
                             glob('launch/*.launch.py')),
                (os.path.join('share', package_name, 'config'),
                             glob('config/*.yaml')),
      ],
      install_requires=['setuptools'],
      zip_safe=True,
      maintainer='Krishna Digamarthi',
      maintainer_email='shivasaikrishna23@gmail.com',
      description='Side-scan sonar crab-pot detection with ROS 2',
      license='MIT',
      tests_require=['pytest'],
      entry_points={
                'console_scripts': [
                              'packet_stitcher = sonar_perception.packet_stitcher:main',
                              'detector        = sonar_perception.detector:main',
                              'gnss_imu_fusion = sonar_perception.gnss_imu_fusion:main',
                ],
      },
)
