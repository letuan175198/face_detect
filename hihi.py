from __future__ import print_function
import sys
import cv2
from random import randint

if __name__ == '__main__':
    # đường dẫn đến  videop
    videoPath = "videos/3.mp4"
    # tạo đối tượng chứa nội dung video đọc từ file lên
    cap = cv2.VideoCapture(videoPath)
    # đọc frame đầu tiên của video
    success, frame = cap.read()
    # n = 99
    # for i in range(n):
    #     success, frame = cap.read()
    # kiểm tra xem có thành công không , nếu không thì thoát ra
    if not success:
        print('Failed to read video')
        sys.exit(1)

    ## chuyển ảnh thành màu xám
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ## tải trình phân loại khuôn mặt đã được đào tạo trong opencv
    haar_cascade_face = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    ## lấy các hình chữ nhật có khuôn mặt được phát hiện , đầu vào của hàm này là một ảnh xám
    bboxes = haar_cascade_face.detectMultiScale(frame_gray, scaleFactor=1.08, minNeighbors=5 , minSize=(30,30))
    ## tạo một mảng để chứa màu cho các đối tượng
    colors = []

    # Tạo đối tượng để chứa các tracker
    multiTracker = cv2.MultiTracker_create()

    # Khởi tạo các tracker cho từng đối tượng mà trình phát hiện đã phát hiện ra
    for bbox_ in bboxes:
        # vì đầu vào của hàm dưới là tuple nên phải chuyển ndarray thành tuple
        bbox = (bbox_[0],bbox_[1],bbox_[2],bbox_[3])
        colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
        tracker = cv2.TrackerCSRT_create()
        multiTracker.add(tracker, frame, bbox)

    # for i, newbox in enumerate(bboxes):
    #     p1 = (int(newbox[0]), int(newbox[1]))
    #     p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
    #     cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
    # from matplotlib import pyplot as plt
    # plt.imshow(frame)
    # plt.show()

    while True :
        # lấy một khung mới để tracking
        success, frame = cap.read()
        if not success:
            print("Không lấy được thêm khung mới")
            break

        # cập nhật vị trí của đối tượng với một khung mới
        success, boxes = multiTracker.update(frame)
        if not success:
            print("Không tìm thấy đối tượng trong khung mới")
            #continue
        # vẽ hình bao cho đối tượng
        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

        # show frame
        cv2.imshow('MultiTracker', frame)

        # Nhấn dấu cách để kết thúc
        k = cv2.waitKey(50) & 0xFF
        if k == 32:
            break

