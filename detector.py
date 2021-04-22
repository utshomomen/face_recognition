import cv2
from tkinter import *
import numpy as np
import pickle
import os
import scipy

import sqlite3
import sqlite3database


def EXIT():

     return exit()
# comment of login


def LOGIN():

    buttonCam = Button(root, text='FACE RECOGNITION', command = RECOGNITION)
    #buttonCAM.pack(side=DOWN)
    buttonCam.grid(row=120 , column = 5)
    #print ('Clicked')

    button3 = Button(root,text = 'Vcamera' , command = CAMERATEST)
    button3.grid(row =70 , column =6)
        # button3.pack(side=RIGHT)

    button4 = Button(root,text = 'Vrecord' , command = VIDERRECORD)
    button4.grid(row =200 , column =3)





recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")
id= 0
cascadePath = ("haarcascade_frontalface_default.xml")
faceCascade = cv2.CascadeClassifier(cascadePath);

def VIDERRECORD():

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    filename = 'video1.avi'
    frames_per_second = 24.0
    res = '720p'


    def change_res(cap, width, height):
        cap.set(3, width)
        cap.set(4, height)

    # Standard Video Dimensions Sizes
    STD_DIMENSIONS = {
        "480p": (640, 480),
        "720p": (1280, 720),
        "1080p": (1920, 1080),
        "4k": (3840, 2160),
    }

    # grab resolution dimensions and set video capture to it.
    def get_dims(cap, res='1080p'):
        width, height = STD_DIMENSIONS["480p"]
        if res in STD_DIMENSIONS:
            width, height = STD_DIMENSIONS[res]
        ## change the current caputre device
        ## to the resulting resolution
        change_res(cap, width, height)
        return width, height

    #
    VIDEO_TYPE = {
        'avi': cv2.VideoWriter_fourcc(*'XVID'),
        # 'mp4': cv2.VideoWriter_fourcc(*'H264'),
        'mp4': cv2.VideoWriter_fourcc(*'XVID'),
    }

    def get_video_type(filename):
        filename, ext = os.path.splitext(filename)
        if ext in VIDEO_TYPE:
            return VIDEO_TYPE[ext]
        return VIDEO_TYPE['avi']

    cap = cv2.VideoCapture(0)
    dims = get_dims(cap, res)
    video_type_cv2 = get_video_type(filename)

    out = cv2.VideoWriter(filename, video_type_cv2, frames_per_second, dims)  # width and height

    def make_1080p():
        cap.set(3, 1920)
        cap.set(4, 1080)

    def make_720p():
        cap.set(3, 1280)
        cap.set(4, 720)

    def make_480p():
        cap.set(3, 640)
        cap.set(4, 480)

    def change_res(width, height):
        cap.set(3, width)
        cap.set(4, height)

    make_480p()
    change_res(720, 480)

    while True:

        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        out.write(frame)

        #gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        #faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                            #  flags=cv2.CASCADE_SCALE_IMAGE)  # flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
        # show the frame

        cv2.imshow('frame', frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

            # while everything is done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def CAMERATEST():
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    while (True):
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()











def RECOGNITION():





 def getProfile(id):
    conn = sqlite3.connect("FaceApp.db")
    query = "SELECT * FROM People WHERE ID=" + str(id)
    cursor = conn.execute(query)
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile




 cap = cv2.VideoCapture(0)
 while True:
       ret, img = cap.read()
       fontface = cv2.FONT_HERSHEY_SIMPLEX
       fontscale = 1
       fontcolor = (255, 255, 255)

       locy = int(img.shape[0] / 2)  # the text location will be in the middle
       locx = int(img.shape[1] / 2)  #

       gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
       faces=faceCascade.detectMultiScale(gray, 1.2,5)
       for(x,y,w,h) in faces:
           cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
           id, conf = recognizer.predict(gray[y:y+h,x:x+w])
           if conf < 80:
               profile = getProfile(id)
               if (profile != None):
                   cv2.putText(img, "Name: " + str(profile[1]), (x, y + h + 30), fontscale, 0.4, (0, 0, 255), 1);
                   cv2.putText(img, "Age: " + str(profile[2]), (x, y + h + 50), fontscale, 0.4, (0, 0, 255), 1);
                   cv2.putText(img, "Gender: " + str(profile[3]), (x, y + h + 70), fontscale, 0.4, (0, 0, 255), 1);
                   cv2.putText(img, "Birthday: " + str(profile[4]), (x, y + h + 90), fontscale, 0.4,(0, 0, 255), 1);
               else:
                   cv2.putText(img, "Name: Utsho", (x, y + h + 30), fontscale, 0.4, (0, 0, 255), 1);
                   cv2.putText(img, "Age: 20", (x, y + h + 50), fontscale, 0.4, (0, 0, 255), 1);
                   cv2.putText(img, "Gender: m", (x, y + h + 70), fontscale, 0.4, (0, 0, 255), 1);
                   cv2.putText(img, "Birthday: 99", (x, y + h + 90), fontscale, 0.4, (0, 0, 255), 1);
         #  if(conf<50):
           #    if(id==1):
             #      id="hsshhs"
          #     elif(id==2):
           #        id="Sam"
           #else:
          # id="Unknown motherfucker"
           #if (id == 1):
            #      id="utsho"
            #   elif(id == 2):
               #  id = "emma watson {0:.2f}%".format(round(100 - conf, 2))
            #elif(id==2):
            #id = "Emma watson {0:.2f}%".format(round(100 - conf, 2))
               #cv2.cv.PutText(cv2.cv.fromarray(img),str(id), (x,y+h), 255)

               cv2.putText(img, str(id), (x, y + h), fontface, fontscale, fontcolor)
               cv2.imshow('frame', img)
               if(cv2.waitKey(1) & 0xFF == ord('q')):
                 break


                 cap.release()
                 cv2.destroyAllWindows()











root = Tk()
window = root.geometry("700x500")
root.title("Face App")
root.iconbitmap('logo.ico')

label1 = Label(root, text="Name:", bg='red', width=50)
label1.grid(row=0, column=0, sticky=E)

label2 = Label(root, text='Login:', bg='red', fg='black', width=50)
label2.grid(row=1, column=0, sticky=E)

entry1 = Entry(root)
entry1.grid(row=0, column=10, ipadx=6)
entry1.insert(25, 'FACE APP')

entry2 = Entry(root)
entry2.grid(row=1, column=10, ipadx=6)
entry2.insert(25, 'utshomomen')

button1 = Button(root, text='OK', command=LOGIN)
button1.grid(row=10, column=0, ipadx=2)

button2 = Button(root, text='Quit', command=EXIT)
button2.grid(row=10, column=12, ipadx=2)

prompt = Label(text="\nWelcome to FACE APP !", font=("Helvetica", 8))
prompt.grid( row= 0, column=100,ipadx=2 )

root.mainloop()