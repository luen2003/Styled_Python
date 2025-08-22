import cv2
import mediapipe as mp
import time
import os
import numpy as np
import random
import sys

# Xử lý đường dẫn khi chạy exe
def resource_path(relative_path):
    """Trả về đường dẫn thực tế tới file khi chạy exe (PyInstaller)"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# Tải ảnh từ thư mục phụ kiện
def load_images(folder):
    images = []
    for f in sorted(os.listdir(folder)):
        if f.endswith('.png'):
            img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_UNCHANGED)
            if img is not None:
                images.append(img)
    print(f"✅ Loaded {len(images)} images from {folder}")
    return images

# Hàm chèn ảnh có alpha vào frame
def overlay_transparent(background, overlay, x, y, scale=1):
    if overlay is None:
        return background

    h, w = overlay.shape[:2]
    overlay = cv2.resize(overlay, (int(w * scale), int(h * scale)))
    h, w = overlay.shape[:2]

    if overlay.shape[2] == 3:
        alpha_channel = 255 * np.ones((h, w), dtype=overlay.dtype)
        overlay = np.dstack([overlay, alpha_channel])

    if x < 0:
        overlay = overlay[:, -x:]
        w += x
        x = 0
    if y < 0:
        overlay = overlay[-y:, :]
        h += y
        y = 0
    if x + w > background.shape[1]:
        overlay = overlay[:, :background.shape[1] - x]
        w = background.shape[1] - x
    if y + h > background.shape[0]:
        overlay = overlay[:background.shape[0] - y, :]
        h = background.shape[0] - y

    if w <= 0 or h <= 0:
        return background

    alpha = overlay[:, :, 3] / 255.0
    for c in range(3):
        background[y:y+h, x:x+w, c] = (
            background[y:y+h, x:x+w, c] * (1 - alpha) +
            overlay[:, :, c] * alpha
        )
    return background

# Load phụ kiện từ đường dẫn resource
hats = load_images(resource_path("assets/hats"))
bows = load_images(resource_path("assets/bows"))
candies = load_images(resource_path("assets/candies"))
beards = load_images(resource_path("assets/beards"))

if not hats or not bows or not candies or not beards:
    print("❌ Thiếu ảnh trong thư mục assets/. Vui lòng kiểm tra lại.")
    exit()

# MediaPipe setup
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Mở webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Không thể mở webcam.")
    exit()

# Cấu hình ghi video .mp4
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

# Biến điều khiển
interval = 0.2
last_switch = time.time()
hat_idx = bow_idx = candy_idx = beard_idx = 0

print("✅ Chạy thành công. Nhấn phím X hoặc đóng cửa sổ để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]
        landmarks = face.landmark

        x_hat = int(landmarks[10].x * w) - 169
        y_hat = int(landmarks[10].y * h) - 190

        x_bow = int(landmarks[152].x * w) - 80
        y_bow = int(landmarks[152].y * h)

        x_candy = int(landmarks[13].x * w) - 30
        y_candy = int(landmarks[13].y * h) - 10

        x_beard = int(landmarks[13].x * w) - 50
        y_beard = int(landmarks[13].y * h) - 60

        frame = overlay_transparent(frame, hats[hat_idx], x_hat, y_hat, scale=1.4)
        frame = overlay_transparent(frame, bows[bow_idx], x_bow, y_bow, scale=1.0)
        frame = overlay_transparent(frame, candies[candy_idx], x_candy, y_candy, scale=0.5)
        frame = overlay_transparent(frame, beards[beard_idx], x_beard, y_beard, scale=0.5)

    if time.time() - last_switch > interval:
        hat_idx = random.randint(0, len(hats) - 1)
        bow_idx = random.randint(0, len(bows) - 1)
        candy_idx = random.randint(0, len(candies) - 1)
        beard_idx = random.randint(0, len(beards) - 1)
        print(f"Đổi phụ kiện: mũ {hat_idx}, nơ {bow_idx}, kẹo {candy_idx}, râu {beard_idx}")
        last_switch = time.time()

    out.write(frame)
    cv2.imshow("TikTok Face Filter", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('x') or key == ord('X'):
        print("👋 Thoát chương trình.")
        break

    if cv2.getWindowProperty("TikTok Face Filter", cv2.WND_PROP_VISIBLE) < 1:
        print("👋 Đã đóng cửa sổ, thoát chương trình.")
        break

cap.release()
out.release()
cv2.destroyAllWindows()
