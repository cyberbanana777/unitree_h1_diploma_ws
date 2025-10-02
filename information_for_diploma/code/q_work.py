import pyrealsense2 as rs
import numpy as np
import cv2

# Настройка конвейера RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Создаем объект для выравнивания глубины к цвету
align_to = rs.stream.color
align = rs.align(align_to)

# Старт потока
profile = pipeline.start(config)

# Получаем масштаб глубины (важно для корректного отображения)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

# Создаем объект для фильтрации глубины (уменьшаем шум)
spatial = rs.spatial_filter()
spatial.set_option(rs.option.holes_fill, 3)  # Агрессивное заполнение "дыр"

try:
    while True:
        # Ждем кадры
        frames = pipeline.wait_for_frames()
        
        # Выравниваем кадры глубины к цветному
        aligned_frames = align.process(frames)
        
        # Получаем выровненные кадры
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not depth_frame or not color_frame:
            continue
            
        # Применяем фильтр к кадру глубины
        filtered_depth = spatial.process(depth_frame)
        
        # Конвертируем в numpy массивы
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(filtered_depth.get_data())
        
        # 1. СЕГМЕНТАЦИЯ (Детекция по цвету в HSV)
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Улучшаем маску
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        # Находим контуры для визуализации
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 2. ПОЛУЧЕНИЕ ОБЛАКА ТОЧЕК
        pc = rs.pointcloud()
        pc.map_to(color_frame)  # Привязываем текстуры к цветному кадру
        points = pc.calculate(filtered_depth)  # Используем отфильтрованный кадр глубины
        
        # Получаем вершины и текстуры облака точек
        vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        
        # 3. ФИЛЬТРАЦИЯ ОБЛАКА ТОЧЕК ПО МАСКЕ
        mask_flat = mask.ravel().astype(bool)
        
        # Убеждаемся, что количество точек соответствует размеру маски
        if len(vtx) != mask_flat.size:
            print("Ошибка: размер облака точек не соответствует размеру маски")
            continue
            
        obj_points = vtx[mask_flat]
        
        # 4. РАСЧЕТ ЦЕНТРА И ОТРИСОВКА
        if len(obj_points) > 100:  # Добавляем минимальный порог точек
            # Используем медиану для устойчивости к выбросам
            center_3d = np.median(obj_points, axis=0)
            
            # Отображаем позицию на изображении
            cv2.putText(color_image, f"X: {center_3d[0]:.2f}m", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(color_image, f"Y: {center_3d[1]:.2f}m", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(color_image, f"Z: {center_3d[2]:.2f}m", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Отрисовываем контур и центр на цветном изображении
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                cv2.drawContours(color_image, [largest_contour], -1, (0, 255, 0), 2)
                
                # Находим центр контура в 2D
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.circle(color_image, (cX, cY), 5, (0, 255, 0), -1)
        else:
            cv2.putText(color_image, "Object not found", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Применяем цветовую карту к глубине для визуализации
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET
        )
        
        # Накладываем маску на цветное изображение для наглядности
        color_with_mask = cv2.bitwise_and(color_image, color_image, mask=mask)
        
        # Отрисовка для отладки
        cv2.imshow('RGB', color_image)
        cv2.imshow('Depth', depth_colormap)
        cv2.imshow('Mask', mask)
        cv2.imshow('Color with Mask', color_with_mask)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()