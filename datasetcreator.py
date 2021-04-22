import numpy as np
import cv2
import sqlite3


detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)


def database(id ,name):
    #conn = sqlite3.connect('FaceApp')
    conn = sqlite3.connect('FaceApp.db')
    query = "SELECT * FROM People WHERE ID=" + str(id)
    cursor=conn.execute(query)
    isRecordExist=0
    for row in cursor:
        isRecordExist=1
        if isRecordExist==1:
            query="UPDATE People SET Name="+str(name)+" WHERE ID="+str(id)
        else:
            query="INSERT INTO people(ID,Name) Values("+str(id)+",' "+str(name)+" ' )"

            conn.execute(query)
            conn.commit()
            conn.close()

id= input('enter your id')
name=input('enter your name')
database(id,name)
sampleNum=0;
while (True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        sampleNum=sampleNum+1;
        cv2.imwrite("dataset/User." +str(id) + "." +str(sampleNum)+ ".jpg", gray[y:y+h,x:x+w])

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)



    cv2.imshow('frame', img)
    cv2.waitKey(100);
    if(sampleNum>20):
        break

cap.release()
cv2.destroyAllWindows()