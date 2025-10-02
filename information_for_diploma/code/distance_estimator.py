#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from message_filters import TimeSynchronizer, Subscriber

class DistanceEstimator(Node):
    def __init__(self):
        super().__init__('distance_estimator')
        
        # Мост для преобразования изображений
        self.bridge = CvBridge()
        
        # Параметры для детекции по цвету (можно вынести в параметры ROS)
        self.lower_color = np.array([100, 50, 50])   # HSV - нижняя граница синего
        self.upper_color = np.array([140, 255, 255]) # HSV - верхняя граница синего
        
        # Подписка на синхронизированные изображения
        rgb_sub = Subscriber(self, Image, 'camera/color/image_raw')
        depth_sub = Subscriber(self, Image, 'camera/depth/image_raw')
        
        # Синхронизатор сообщений
        self.ts = TimeSynchronizer([rgb_sub, depth_sub], 10)
        self.ts.registerCallback(self.image_callback)
        
        # Публикатор для позиции объекта
        self.position_publisher = self.create_publisher(PointStamped, 'object_position', 10)
        
        self.get_logger().info('Distance estimator node started')
    
    def image_callback(self, rgb_msg, depth_msg):
        try:
            # Преобразование ROS сообщений в OpenCV изображения
            color_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')
            
            # 1. СЕГМЕНТАЦИЯ (Детекция по цвету в HSV)
            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
            
            # Улучшаем маску
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            
            # 2. НАХОЖДЕНИЕ КОНТУРОВ
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Находим наибольший контур
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Создаем маску только для наибольшего контура
                contour_mask = np.zeros_like(mask)
                cv2.drawContours(contour_mask, [largest_contour], -1, 255, -1)
                
                # 3. ВЫЧИСЛЕНИЕ ПОЗИЦИИ ОБЪЕКТА
                # Получаем координаты всех точек в контуре
                y, x = np.where(contour_mask == 255)
                
                if len(x) > 0 and len(y) > 0:
                    # Вычисляем медиану координат
                    center_x = int(np.median(x))
                    center_y = int(np.median(y))
                    
                    # Получаем глубину в центре объекта
                    depth_value = depth_image[center_y, center_x] / 1000.0  # преобразуем в метры
                    
                    # Публикуем позицию объекта
                    position_msg = PointStamped()
                    position_msg.header.stamp = rgb_msg.header.stamp
                    position_msg.header.frame_id = rgb_msg.header.frame_id
                    
                    # Для преобразования пиксельных координат в 3D точку нужна калибровка камеры
                    # Здесь используем упрощенный подход (предполагаем центральную точку)
                    # В реальном приложении нужно использовать параметры камеры из CameraInfo
                    position_msg.point.x = depth_value
                    position_msg.point.y = (center_x - 320) * depth_value / 600.0  # примерное преобразование
                    position_msg.point.z = (center_y - 240) * depth_value / 600.0  # примерное преобразование
                    
                    self.position_publisher.publish(position_msg)
                    
                    # Логируем информацию
                    self.get_logger().info(
                        f'Object detected at: X: {position_msg.point.x:.2f}m, '
                        f'Y: {position_msg.point.y:.2f}m, Z: {position_msg.point.z:.2f}m'
                    )
            
            # Для отладки: отображаем изображения
            if self.get_parameter('debug').value:
                # Рисуем контуры на оригинальном изображении
                cv2.drawContours(color_image, contours, -1, (0, 255, 0), 2)
                
                # Отображаем центр объекта
                if 'center_x' in locals() and 'center_y' in locals():
                    cv2.circle(color_image, (center_x, center_y), 5, (0, 0, 255), -1)
                
                # Показываем изображения
                cv2.imshow('RGB', color_image)
                cv2.imshow('Mask', mask)
                cv2.waitKey(1)
                
        except Exception as e:
            self.get_logger().error(f'Error in image processing: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = DistanceEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
