#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
from tf2_geometry_msgs import do_transform_point
import time

class Object3DLocator(Node):
    def __init__(self):
        super().__init__('object_3d_locator')
        
        # Инициализация
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Параметры с правильными значениями по умолчанию
        self.declare_parameter('camera_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('pointcloud_topic', '/camera/points')
        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('camera_frame', 'camera_depth_optical_frame')
        
        # Получение параметров
        camera_topic = self.get_parameter('camera_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        pointcloud_topic = self.get_parameter('pointcloud_topic').value
        
        # Подписки
        self.image_sub = self.create_subscription(
            Image, camera_topic, self.image_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, depth_topic, self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self.camera_info_callback, 10)
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, pointcloud_topic, self.pointcloud_callback, 10)
        
        # Публикатор для визуализации
        self.point_pub = self.create_publisher(PointStamped, '/object_3d_point', 10)
        
        # Переменные для хранения данных
        self.current_image = None
        self.current_depth = None
        self.camera_info = None
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.data_ready = False
        
        # Время последнего получения сообщений
        self.last_image_time = 0
        self.last_depth_time = 0
        self.last_info_time = 0
        
        # Таймер для проверки доступности данных
        self.check_timer = self.create_timer(1.0, self.check_data_availability)
        
        self.get_logger().info('Object 3D Locator node initialized')
        self.get_logger().info('Click on the image to get 3D coordinates')
        self.get_logger().info(f'Subscribed to: {camera_topic}, {depth_topic}, {camera_info_topic}')
    
    def check_data_availability(self):
        """Проверка доступности всех необходимых данных"""
        current_time = time.time()
        
        # Проверяем, не устарели ли данные (больше 2 секунд)
        image_ok = self.current_image is not None and (current_time - self.last_image_time) < 2.0
        depth_ok = self.current_depth is not None and (current_time - self.last_depth_time) < 2.0
        info_ok = self.camera_matrix is not None and (current_time - self.last_info_time) < 2.0
        
        if image_ok and depth_ok and info_ok:
            if not self.data_ready:
                self.data_ready = True
                self.get_logger().info('All data is available and ready for processing')
        else:
            missing = []
            if not image_ok: missing.append('image')
            if not depth_ok: missing.append('depth')
            if not info_ok: missing.append('camera_info')
            self.get_logger().warn(f'Waiting for: {", ".join(missing)}')
            
            # Сбрасываем флаг готовности
            self.data_ready = False
    
    def camera_info_callback(self, msg):
        """Получение параметров камеры"""
        try:
            self.camera_info = msg
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.distortion_coeffs = np.array(msg.d)
            self.last_info_time = time.time()
            
            if self.camera_matrix is not None:
                self.get_logger().info('Camera parameters received')
                self.get_logger().info(f'Camera matrix: {self.camera_matrix[0, 0]:.1f}, {self.camera_matrix[1, 1]:.1f}')
                
        except Exception as e:
            self.get_logger().error(f'Error processing camera info: {e}')
    
    def image_callback(self, msg):
        """Обработка цветного изображения"""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.last_image_time = time.time()
            
            # Показываем изображение для выбора точки
            cv2.imshow('Click on object', self.current_image)
            cv2.setMouseCallback('Click on object', self.mouse_callback)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
    
    def depth_callback(self, msg):
        """Обработка изображения глубины"""
        try:
            self.current_depth = self.bridge.imgmsg_to_cv2(msg, '16UC1')
            self.last_depth_time = time.time()
            
            # Логируем информацию о глубине
            if self.current_depth is not None:
                non_zero = np.count_nonzero(self.current_depth)
                total = self.current_depth.size
                self.get_logger().info(f'Depth image: {non_zero}/{total} non-zero pixels ({non_zero/total*100:.1f}%)', 
                                      throttle_duration_sec=5.0)
                
        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {e}')
    
    def pointcloud_callback(self, msg):
        """Альтернативный метод: получение точки из облака точек"""
        # Этот метод может быть более точным для некоторых случаев
        pass
    
    def mouse_callback(self, event, u, v, flags, param):
        """Обработка клика мыши по изображению"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.data_ready:
                self.get_logger().warn('Data not ready yet. Please wait for all topics...')
                self.check_data_availability()  # Принудительная проверка
                return
            
            self.get_logger().info(f'Clicked at pixel coordinates: u={u}, v={v}')
            self.calculate_3d_coordinates(u, v)
    
    def calculate_3d_coordinates(self, u, v):
        """Вычисление 3D координат по пикселю"""
        if not self.data_ready:
            self.get_logger().warn('Data not ready yet')
            return
        
        try:
            # Проверяем границы изображения
            if self.current_depth is None:
                self.get_logger().warn('No depth image available')
                return
                
            height, width = self.current_depth.shape
            if u < 0 or u >= width or v < 0 or v >= height:
                self.get_logger().warn(f'Coordinates out of bounds: u={u}, v={v}, image size: {width}x{height}')
                return
            
            # Получаем значение глубины (в мм)
            depth_value = self.current_depth[v, u]
            
            if depth_value == 0:
                self.get_logger().warn('No depth data at this point (depth=0)')
                # Попробуем найти ближайшую точку с данными глубины
                depth_value = self.find_nearest_depth(u, v, 10)
                if depth_value == 0:
                    self.get_logger().warn('Could not find depth data nearby')
                    return
                self.get_logger().info(f'Using depth from nearby point: {depth_value}mm')
            
            # Конвертируем в метры
            Z = depth_value / 1000.0  # Из мм в метры
            
            # Получаем параметры камеры
            fx = self.camera_matrix[0, 0]  # Фокусное расстояние по x
            fy = self.camera_matrix[1, 1]  # Фокусное расстояние по y
            cx = self.camera_matrix[0, 2]  # Главная точка по x
            cy = self.camera_matrix[1, 2]  # Главная точка по y
            
            # Вычисляем 3D координаты относительно камеры
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            
            self.get_logger().info(f'Camera coordinates: X={X:.3f}m, Y={Y:.3f}m, Z={Z:.3f}m')
            
            # Публикуем точку в системе камеры
            self.publish_camera_point(X, Y, Z)
            
            # Преобразуем в целевую систему координат
            self.transform_to_target_frame(X, Y, Z)
            
        except Exception as e:
            self.get_logger().error(f'Error calculating 3D coordinates: {e}')
    
    def find_nearest_depth(self, u, v, radius=10):
        """Поиск ближайшей точки с данными глубины"""
        height, width = self.current_depth.shape
        
        for r in range(1, radius + 1):
            for du in range(-r, r + 1):
                for dv in range(-r, r + 1):
                    nu, nv = u + du, v + dv
                    if (0 <= nu < width and 0 <= nv < height and 
                        self.current_depth[nv, nu] != 0):
                        return self.current_depth[nv, nu]
        return 0
    
    def publish_camera_point(self, x, y, z):
        """Публикация точки в системе координат камеры"""
        point_msg = PointStamped()
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.header.frame_id = self.get_parameter('camera_frame').value
        point_msg.point.x = x
        point_msg.point.y = y
        point_msg.point.z = z
        
        self.point_pub.publish(point_msg)
        self.get_logger().info(f'Published point: ({x:.3f}, {y:.3f}, {z:.3f})')
    
    def transform_to_target_frame(self, x, y, z):
        """Преобразование координат в целевую систему отсчета"""
        try:
            # Создаем точку в системе камеры
            point_in_camera = PointStamped()
            point_in_camera.header.stamp = self.get_clock().now().to_msg()
            point_in_camera.header.frame_id = self.get_parameter('camera_frame').value
            point_in_camera.point.x = x
            point_in_camera.point.y = y
            point_in_camera.point.z = z
            
            # Получаем трансформацию
            target_frame = self.get_parameter('target_frame').value
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                point_in_camera.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            
            # Применяем трансформацию
            point_in_target = do_transform_point(point_in_camera, transform)
            
            self.get_logger().info(
                f'Target coordinates ({target_frame}): '
                f'X={point_in_target.point.x:.3f}m, '
                f'Y={point_in_target.point.y:.3f}m, '
                f'Z={point_in_target.point.z:.3f}m'
            )
            
        except tf2_ros.LookupException:
            self.get_logger().error('Transform lookup failed. Check if TF is being published.')
        except tf2_ros.ConnectivityException:
            self.get_logger().error('Transform connectivity failed')
        except tf2_ros.ExtrapolationException:
            self.get_logger().error('Transform extrapolation failed')
        except Exception as e:
            self.get_logger().error(f'Transform error: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = Object3DLocator()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()