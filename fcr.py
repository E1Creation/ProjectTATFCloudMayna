import cv2
import numpy as np
import tensorflow as tf
import mysql_connector as sq
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from flask import render_template
from datetime import datetime
from statistics import mode
from firestore import markAttendanceIntoCloud

# Load model

# Settings
size = (224, 224)


# Presence System   
def markAttendanceIntoDB(name,id):
    status="masuk"
    now = datetime.now()
    dtString = now.strftime('%H:%M:%S')
    dateString=now.strftime('%d-%m-%y')
    sq.insertRow(id,dateString,name,dtString,status)

# Doing some Face Recognition with the webcam
def normalizeImage(img):
    img = img/255
    return img



# Defining a function that will do the detections
def recognition(gray, frame, model,face_cascade,listName,listId,namaLabel):

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        frameSize = frame[y:y+h, x:x+w]
      #  frameCrop=cv2.resize(np.float32(frameCrop),size)
        frameCrop = cv2.cvtColor(frameSize, cv2.COLOR_BGR2RGB)
        frameCrop=cv2.resize(np.float32(frameCrop),(224,224), interpolation=cv2.INTER_AREA)
        frameCrop= image.img_to_array(frameCrop)
        frameCrop= np.expand_dims(frameCrop, axis=0)
        images = normalizeImage(frameCrop)
        #images = preprocess_input(x)
        #images = inception(x)
        #images=vggpre(x)
        result = model.predict(images)
        nmax = np.argmax(result)
        nama = namaLabel[nmax]
      #  print(nmax, nama)
      #  print(len(frameSize))
        
        if len(frameSize) >= 150:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, nama, (x,y+h+20), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0))
            listName.append(nama)
            listId.append(nmax)
            print(listName)
            if len(listName) >= 21:
                print("berhasil disimpan")
                markAttendanceIntoCloud(str(mode(listName)))
                markAttendanceIntoDB(str(mode(listName)),str(mode(listId)))
                listName.clear()
                listId.clear()
        #    cv2.putText(frame, str(akurasi),(x,y-2),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255))
    return frame

def gen_frame():
    listName = []
    listId = []
    face_cascade = cv2.CascadeClassifier('D:\Dataku\TA\ProjectTA\projekTAGoogleColab\haarcascade_frontalface_default.xml')
    namaLabel = ['aghus sofwan', 'alam suminto', 'bari', 'bondan', 'enda wista sinuraya', 'gede ananda adi apriliawan', 'gusti ngurah bagus amarry krisna', 'i kadek dwi gita purnama pramudya', 'i made dwiki satria wibawa', 'i made parasya maharta', 'i putu augi oka adiana', 'i putu kaesa wahyu prana aditya', 'kurniawan sudirman', 'mochammad facta', 'muhammad ilham maulana', 'munawar agus riyadi', 'putu irianti putri astari', 'putu kerta adi pande', 'wahyul amien syafei', 'yosua alvin adi a']
    new_model = tf.keras.models.load_model('D:\Dataku\TA\ProjectTA\projekTAGoogleColab\model_inference_20_terbaik_13_VGG19.h5')

 #   print("model inference berhasil diinput")
   # video_capture = cv2.VideoCapture(0)
#    video_capture = cv2.VideoCapture('http://192.168.1.2:8080/videofeed')
    video_capture = cv2.VideoCapture('http://192.168.43.1:8080/videofeed')
    video_capture.set(3, 1280)
    video_capture.set(4, 720)
    while True:
        try:
                _, frame = video_capture.read()
                if _ and frame is not None:
                # print(len(frame))
                    gray = cv2.cvtColor(np.array(frame, dtype = 'uint8'), cv2.COLOR_BGR2GRAY)
                    canvas= recognition(gray, frame, new_model,face_cascade,listName,listId,namaLabel)
                   # cv2.imshow('Video', canvas)
                    #cv2.imshow('videCrop',frameCrop)
                if not _:
                    break
                else:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frameWeb = buffer.tobytes()
                    yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frameWeb + b'\r\n')  # concat frame one by one and show result
        except Exception as e:
            print(str(e))    
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()