import cv2
import numpy as np

FACE_PROTO = "D:/gad/opencv_face_detector.pbtxt"
FACE_MODEL = "D:/gad/opencv_face_detector_uint8.pb"

AGE_PROTO = "D:/gad/age_deploy.prototxt"
AGE_MODEL = "D:/gad/age_net.caffemodel"

GENDER_PROTO = "D:/gad/gender_deploy.prototxt"
GENDER_MODEL = "D:/gad/gender_net.caffemodel"

faceNet = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
ageNet = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
genderNet = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)

AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300),
                                 [104, 117, 123], swapRB=False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2),
                          (0, 255, 0), 2, 8)
    return frameOpencvDnn, faceBoxes

frame = cv2.imread("D:/gad/your_image.jpg")
resultImg, faceBoxes = highlightFace(faceNet, frame)

for faceBox in faceBoxes:
    face = frame[max(0, faceBox[1]-10):min(faceBox[3]+10, frame.shape[0]-1),
                 max(0, faceBox[0]-10):min(faceBox[2]+10, frame.shape[1]-1)]

    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                 [78.4263377603, 87.7689143744, 114.895847746],
                                 swapRB=False)

    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    gender = GENDER_LIST[genderPreds[0].argmax()]

    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    age = AGE_LIST[agePreds[0].argmax()]

    label = f"{gender}, {age}"
    cv2.putText(resultImg, label, (faceBox[0], faceBox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

cv2.imshow("Gender and Age Detection", resultImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
