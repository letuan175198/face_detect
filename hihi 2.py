from __future__ import print_function
import sys
import cv2
from random import randint
import  math

def compare(b1,b2):
    x1 = b1[0] + b1[2]*0.5
    y1 = b1[1] + b1[3]*0.5
    x2 = b2[0] + b2[2] * 0.5
    y2 = b2[1] + b2[3] * 0.5
    d = math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
    if(d<50):
        return True
    else:
        return False
def check_exist(b2,bbox):
    for b1 in bbox:
        if compare(b1,b2):
            return True
    return False

if __name__ == '__main__':

    # tạo đối tượng chứa nội dung video đọc từ camera
    cap = cv2.VideoCapture(0)
    # đọc frame đầu tiên của video
    success, frame = cap.read()
    # kiểm tra xem có thành công không , nếu không thì thoát ra
    if not success:
        print('Failed to read video')
        sys.exit(1)

    ## chuyển ảnh thành màu xám
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ## tải trình phân loại khuôn mặt đã được đào tạo trong opencv
    haar_cascade_face = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

    scaleFactor = 1.15
    minNeighbors = 5
    minSize = (30, 30)
    ## lấy các hình chữ nhật có khuôn mặt được phát hiện , đầu vào của hàm này là một ảnh xám
    bboxes = haar_cascade_face.detectMultiScale(frame_gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors , minSize=(minSize))
    print("Số khuôn mặt trong khung đầu là : "+str(len(bboxes)))
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

    time = 1
    while True :
        # lấy một khung mới để tracking
        success, frame = cap.read()
        if not success:
            print("Không lấy được thêm khung mới")
            break
        timer = cv2.getTickCount()
        # cập nhật vị trí của đối tượng với một khung mới
        success, boxes = multiTracker.update(frame)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        if not success:
            print("Không tìm thấy đối tượng trong khung mới")
            #continue
        # vẽ hình bao cho đối tượng
        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        ## cập nhật khuôn mặt mới
        bboxs_new = haar_cascade_face.detectMultiScale(frame_gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors , minSize=minSize)
        if(time%5==0):
            for box_2 in bboxs_new:
                if check_exist(box_2,boxes) :
                    continue
                box_new = (box_2[0], box_2[1], box_2[2], box_2[3])
                colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
                tracker = cv2.TrackerCSRT_create()
                multiTracker.add(tracker, frame, box_new)
        time+=1

        #show frame
        cv2.imshow('MultiTracker', frame)

        # Nhấn dấu cách để kết thúc
        k = cv2.waitKey(50) & 0xFF
        if k == 32:
            break

