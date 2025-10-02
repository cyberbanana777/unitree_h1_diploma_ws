import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    # Получаем путь к нашему config-файлу
    config_dir = get_package_share_directory('camera_config_diploma')
    config_file = os.path.join(config_dir, 'config', 'camera_params.yaml')
    
    # Получаем путь к системному launch-файлу realsense2_camera
    realsense_pkg_share = get_package_share_directory('realsense2_camera')
    realsense_launch_file = os.path.join(realsense_pkg_share, 'launch', 'rs_launch.py')
    
    # Устанавливаем переменную окружения
    return LaunchDescription([
        SetEnvironmentVariable(
            name='REALSENSE_CONFIG_FILE',
            value=config_file
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(realsense_launch_file)
        )
    ])