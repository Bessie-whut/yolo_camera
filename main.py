import cv2
from ultralytics import YOLO

# 加载模型
model = YOLO('yolov8n.pt')

# 打开摄像头
cap = cv2.VideoCapture(0)

print("摄像头启动中... 按 'q' 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # 识别画面
    results = model(frame, verbose=False)
    
    # 显示画面
    cv2.imshow("YOLO", results[0].plot())
    
    # 按 q 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()