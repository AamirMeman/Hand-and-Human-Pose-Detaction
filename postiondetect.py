import mediapipe as mp
import cv2 as cv

mpdraw=mp.solutions.drawing_utils
mppose=mp.solutions.pose
pose=mppose.Pose()

cap=cv.VideoCapture(0)

while True:
    success,img=cap.read()
    imgrgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    result = pose.process(imgrgb)

    if result.pose_landmarks:
        mpdraw.draw_landmarks(img, result.pose_landmarks,mppose.POSE_CONNECTIONS)

    cv.imshow('image', img)
    cv.waitKey(1)