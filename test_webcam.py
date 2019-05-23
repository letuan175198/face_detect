import numpy as np
import cv2

# Đầu vào của camera và file xml

face_cascade = cv2.CascadeClassifier("haarcascades\haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

# # Tạo đối tượng để ghi video camera ghi được sau khi thực hiện detect
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (640,480))

# Quá trình này dùng để tạo video thành màu xám, và các bước tính toán dữ liệu của khuôn mặt gồm có faces
while (True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor = 1.1, minNeighbors = 5 , minSize=(24,24))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

    # # Đầu ra của video được ghi lại
    # out.write(frame)

    # Toàn bộ trên đã được phác hoa ra màn hình của bạn
    cv2.imshow('Camera', frame)

    #nhấn dấu cách để kết thúc chương trình
    if cv2.waitKey(1) & 0xFF == 32:
        break

cap.release()
# out.release()
cv2.destroyAllWindows()