from __future__ import print_function
import sys
import cv2
from random import randint
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to vieos need detection")
ap.add_argument("-s", "--scaleFacter", help="float")
ap.add_argument("-m", "--minNeighbors", help="int")
# ap.add_argument("-m2", "--minSize", help="tuple. ex (30, 30)")
# ap.add_argument("-m3", "--maxSize", help="tuple. ex (60, 60)")
args = vars(ap.parse_args())

if args["video"] is None:
    videoPath = 0
else: videoPath = args["video"]
if args["scaleFacter"] is None:
    scaleFacter = 1.08
else: scaleFacter = float(args["scaleFacter"])
if args["minNeighbors"] is None:
    minNeighbors = 8
else: minNeighbors = int(args["minNeighbors"])
# if args["minSize"] is None:
minSize = (30, 30)
# else: minSize = args["minSize"]
# if args["maxSize"] is None:
#maxSize = (40, 40)
maxSize = None
# else: maxSize = args["maxSize"]

if __name__ == '__main__':

    # tạo đối tượng chứa nội dung video đọc từ camera
    cap = cv2.VideoCapture(videoPath)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter('output2.mp4',fourcc, 20.0, (640,480))

    while True:
        success, frame = cap.read()
        # kiểm tra xem có thành công không , nếu không thì thoát ra
        if not success:
            print('Failed to read video')
            sys.exit(1)
        cv2.imshow("Tracking",frame)
        q = cv2.waitKey(1) & 0xff
        if q == 32:
            break
        ## chuyển ảnh thành màu xám
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ## tải trình phân loại khuôn mặt đã được đào tạo trong opencv
        haar_cascade_face = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

        # scaleFacter = 1.15
        # minNeighbors = 5
        # minSize = (30, 30)
        ## lấy các hình chữ nhật có khuôn mặt được phát hiện , đầu vào của hàm này là một ảnh xám
        bboxes = haar_cascade_face.detectMultiScale(frame_gray, scaleFactor=scaleFacter, minNeighbors=minNeighbors , minSize=minSize)
        if(len(bboxes)>=1):
            break
    print("Số khuôn mặt trong khung đầu là : "+str(len(bboxes)))
    ## tạo một mảng để chứa màu cho các đối tượng
    colors = []

    # Tạo đối tượng để chứa các tracker
    tracker = cv2.TrackerCSRT_create()
    bbox_ = bboxes[0]
    # vì đầu vào của hàm dưới là tuple nên phải chuyển ndarray thành tuple
    bbox = (bbox_[0],bbox_[1],bbox_[2],bbox_[3])
    colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
    ok = tracker.init(frame, bbox)

    while True:
        # Read a new frame
        ok, frame = cap.read()
        if not ok:
            break

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Không tracking được", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # hiển thị FPS
        cv2.putText(frame, "FPS : " + str(int(fps)), (50, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 32:
            break

    cap.release()
    # out.release()
    cv2.destroyAllWindows()
