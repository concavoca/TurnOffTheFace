# -*- coding: utf-8 -*-
import cv2
import numpy as np

# Загрузка каскада Хаара для обнаружения лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# Открываем первую попавшуюся камеру (обычно это твоя веб-камера)
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)

# Параметры сглаживания
alpha = 0.2  # Чем меньше, тем плавнее
smooth_center = None
smooth_size = None
fixed_size = 220  # Можно подобрать под себя

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Переводим изображение в оттенки серого для детекции
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Находим лица на кадре
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=8,      # Увеличено для уменьшения ложных срабатываний
        minSize=(120, 120)   # Минимальный размер лица
    )
    profiles = profile_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=8,
        minSize=(120, 120)
    )

    # Поиск профиля на отражённом изображении (для второго направления)
    flipped_gray = cv2.flip(gray, 1)
    flipped_profiles = profile_cascade.detectMultiScale(
        flipped_gray,
        scaleFactor=1.3,
        minNeighbors=8,
        minSize=(120, 120)
    )

    # Собираем все найденные лица (центры и размеры)
    detections = []
    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        size = max(w, h)
        detections.append((center, size))
    for (x, y, w, h) in profiles:
        center = (x + w // 2, y + h // 2)
        size = max(w, h)
        detections.append((center, size))
    frame_width = frame.shape[1]
    for (x, y, w, h) in flipped_profiles:
        x_flipped = frame_width - x - w
        center = (x_flipped + w // 2, y + h // 2)
        size = max(w, h)
        detections.append((center, size))

    # Если есть хотя бы одно лицо, сглаживаем координаты и размер
    if detections:
        # Берём самое крупное лицо (обычно это ваше)
        best = max(detections, key=lambda d: d[1])
        center, size = best
        if smooth_center is None:
            smooth_center = np.array(center, dtype=np.float32)
            smooth_size = float(size)
        else:
            smooth_center = (1 - alpha) * smooth_center + alpha * np.array(center, dtype=np.float32)
            smooth_size = (1 - alpha) * smooth_size + alpha * float(size)
    # Если лиц нет, не обновляем (квадрат останется на месте)

    # Рисуем сглаженный квадрат (почти всегда одного размера)
    if smooth_center is not None:
        # Можно использовать сглаженный размер или фиксированный
        draw_size = int(fixed_size)  # или int(smooth_size)
        center_x, center_y = map(int, smooth_center)
        x1 = max(center_x - draw_size // 2, 0)
        y1 = max(center_y - draw_size // 2, 0)
        x2 = min(center_x + draw_size // 2, frame.shape[1])
        y2 = min(center_y + draw_size // 2, frame.shape[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)

    # Показываем результат
    cv2.imshow('Face Blocker (press Q to quit)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()