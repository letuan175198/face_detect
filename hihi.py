import sys
import cv2
from random import randint
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to vieos need detection")
ap.add_argument("-s", "--scaleFacter", help="float")
ap.add_argument("-m", "--minNeighbors", help="int")
ap.add_argument("-m2", "--minSize", help="tuple. ex (30, 30)")
ap.add_argument("-m3", "--maxSize", help="tuple. ex (60, 60)")
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
maxSize = (40, 40)
# else: maxSize = args["maxSize"]

haar_cascade_face = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
multiTracker = cv2.MultiTracker_create()
cap = cv2.VideoCapture("videos/1.mp4")
colors = []

def read_frame():
    success, frame = cap.read()

    if not success:
        print('Failed to read video')
        sys.exit(1)
    return frame

def detect_face(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bboxes = haar_cascade_face.detectMultiScale(frame_gray, scaleFactor=scaleFacter, minNeighbors=minNeighbors , minSize=minSize, maxSize=maxSize)
    return bboxes

def add_tracker(frame, box):
    colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
    tracker = cv2.TrackerCSRT_create()
    multiTracker.add(tracker, frame, tuple(box))
    return frame

def check_new_face(box1, boxs):
    midpoint = [box1[0] + box1[2]*0.5 , box1[1] + box1[3]*0.5]
    for box2 in boxs:
        if midpoint[0] > box2[0] and midpoint[0] < box2[0] + box2[2]:
            if midpoint[1] > box2[1] and midpoint[1] < box2[1] + box2[3]:
                return False
    return True

def draw(frame, box, ncolor):
    p1 = (int(box[0]), int(box[1]))
    p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
    cv2.rectangle(frame, p1, p2, colors[ncolor], 2, 1)
    return frame

bboxes = None
while True:
    frame = read_frame()
    bboxes = detect_face(frame)
    if len(bboxes) > 0:
        break
for ncolor, box in enumerate(bboxes):
    add_tracker(frame, box)
    draw(frame, box, ncolor)

count = 1
boxes = []
while True :
    count += 1
    frame = read_frame()
    success, boxes = multiTracker.update(frame)
    if not success:
        print("Không tìm thấy đối tượng trong khung mới")

    if count % 10 == 0:
        newboxs = detect_face(frame)
        print (newboxs)
        for i in newboxs:
            if check_new_face(i, boxes):     
                add_tracker(frame, i)
                boxes = np.append(boxes, [i], axis=0)

    for ncolor, box in enumerate(boxes):
        draw(frame, box, ncolor)

    # show frame
    cv2.imshow('MultiTracker', frame)

    # Nhấn dấu cách để kết thúc
    k = cv2.waitKey(50) & 0xFF
    if k == 32:
        break