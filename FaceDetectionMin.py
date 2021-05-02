import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpFace = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils # Draws the detection border
faceDetection = mpFace.FaceDetection()


while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)

    if results.detections :
        for id,detection in enumerate(results.detections):
            #mpDraw.draw_detection(img, detection)
            #print(detection)
            #print(detection.score)
            #print(detection.location_data.relative_bounding_box)

            boundingboxC = detection.location_data.relative_bounding_box
            imgH, imgW, imgC = img.shape
            boundingbox = int(boundingboxC.xmin * imgW), int(boundingboxC.ymin * imgH), \
                          int(boundingboxC.width * imgW), int(boundingboxC.height * imgH)
            cv2.rectangle(img,boundingbox,(255,0,255),2)
            cv2.putText(img,str(int(detection.score[0]*100)),(boundingbox[0],boundingbox[1]-20),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)


    cv2.imshow('Image',img)



    cv2.waitKey(1)