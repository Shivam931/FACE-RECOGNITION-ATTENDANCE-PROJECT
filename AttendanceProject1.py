import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pandas as pd


df = pd.read_csv('BeforeAttendance.csv')

# print(df)


def markAttendance(name):
    with open('BeforeAttendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name in nameList:
            # now = datetime.now()
            # dtString = now.strftime('%H:%M:%S')
            # f.writelines(f'\n{name},{dtString}')
            df.loc[df.Name == f'{name}', 'Presenti '] = '"P"'


path = 'imgAttendance2'
images = []
classNames = []
myList = os.listdir(path)
# print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)
print("encoding complete")


img = face_recognition.load_image_file('test2.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

facesCurFrame = face_recognition.face_locations(img)
encodesCurFrame = face_recognition.face_encodings(img, facesCurFrame)

r = 0
d = 0
for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
    matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
    # print(faceDis)
    matchIndex = np.argmin(faceDis)

    if faceDis[matchIndex] < 0.50:
        name = classNames[matchIndex].upper()
        # print(name)
        y1, x2, y2, x1 = faceLoc
        # y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2+15), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1, y2+11),
                    cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 0), 9)
        cv2.putText(img, name, (x1, y2+11),
                    cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 4)
        markAttendance(name)
        r = r+1
        d = d+1

    else:
        name = 'Unknown'
        # print(name)
        y1, x2, y2, x1 = faceLoc
        # y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2+15), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1, y2+11),
                    cv2.FONT_HERSHEY_COMPLEX, 1.3, (0, 0, 0), 4)
        d = d+1

print("Total number of face detected:", d)
print("Total number of face recognize:", r)
img = cv2.resize(img, (0, 0), None, 0.25, 0.25)
cv2.imshow('Photo', img)
cv2.waitKey(0)
df.to_csv('AfterAttendance.csv')
