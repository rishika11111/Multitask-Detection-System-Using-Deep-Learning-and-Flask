import os
import tensorflow as tf
from dd import *
from flask import Flask, render_template,Response, request, send_from_directory
import cv2
from keras.models import load_model
import numpy as np
from pygame import mixer
import time



app = Flask(__name__, template_folder = 'E:\Projects\Fyrproj\Project')

UPLOAD_FOLDER = r"E:\Projects\Fyrproj\Project"

#COVID DETECTION

cnn_model = tf.keras.models.load_model(r'E:\Projects\Fyrproj\Project\model_covid.h5')

IMAGE_SIZE = 224
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/')
def home():
   return render_template('home.html')

@app.route('/covid_detection')
def upload_file():
   return render_template('covid.html')

def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image /= 255.0  # normalize to [0,1] range
    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

def classify(model, image_path):
    preprocessed_imgage = load_and_preprocess_image(image_path)
    preprocessed_imgage = tf.reshape(
        preprocessed_imgage, (1, IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    prob = cnn_model.predict(preprocessed_imgage)
    label = "Normal " if prob >=0.5 else "Covid Affected "
    label = label+str(prob[[0]])
    return label

@app.route("/covid_output.html", methods=["POST", "GET"])
def upload_files():

    if request.method == "GET":
        return render_template("covid.html")

    else:
        file = request.files["file"]
        upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        print(upload_image_path)
        file.save(upload_image_path)
        label = classify(cnn_model,upload_image_path)
        return render_template(
        "covid_output.html", image_file_name=file.filename, label=label
    )










#DROWSINESS DETECTION


def gen_frames():
    mixer.init()
    sound = mixer.Sound(r'E:\Projects\ML\DrowsinessDetection\alarm.wav')

    face = cv2.CascadeClassifier(r'E:\Projects\ML\HaarCascadeFiles\haarcascade_frontalface_alt.xml')
    leye = cv2.CascadeClassifier(r'E:\Projects\ML\HaarCascadeFiles\haarcascade_lefteye_2splits.xml')
    reye = cv2.CascadeClassifier(r'E:\Projects\ML\HaarCascadeFiles\haarcascade_righteye_2splits.xml')



    lbl=['Close','Open']

    model = load_model('E:\Projects\ML\DrowsinessDetection\models\dd_model.h5')
    path = os.getcwd()

    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    count=0
    score=0
    thicc=2
    rpred=[99]
    lpred=[99]
    cap = cv2.VideoCapture(0)
    while(True):
        ret, frame = cap.read()

        if not ret:
            break
        else:
            height,width = frame.shape[:2]


            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
            left_eye = leye.detectMultiScale(gray)
            right_eye =  reye.detectMultiScale(gray)

            cv2.rectangle(frame, (0,height-50) , (200,height) , color=(0,255,0), thickness=cv2.FILLED )

            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

            for (x,y,w,h) in right_eye:
                r_eye=frame[y:y+h,x:x+w]
                count=count+1
                r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
                r_eye = cv2.resize(r_eye,(24,24))
                r_eye= r_eye/255
                r_eye=  r_eye.reshape(24,24,-1)
                r_eye = np.expand_dims(r_eye,axis=0)
                rpred = model.predict_classes(r_eye)
                if(rpred[0]==1):
                    lbl='Open'
                if(rpred[0]==0):
                    lbl='Closed'
                break

            for (x,y,w,h) in left_eye:
                l_eye=frame[y:y+h,x:x+w]
                count=count+1
                l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)
                l_eye = cv2.resize(l_eye,(24,24))
                l_eye= l_eye/255
                l_eye=l_eye.reshape(24,24,-1)
                l_eye = np.expand_dims(l_eye,axis=0)
                lpred = model.predict_classes(l_eye)
                if(lpred[0]==1):
                    lbl='Open'
                if(lpred[0]==0):
                    lbl='Closed'
                break

            if(rpred[0]==0 and lpred[0]==0):
                score=score+1
                cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
            # if(rpred[0]==1 or lpred[0]==1):
            else:
                score=score-1
                cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1)


            if(score<0):
                score=0
            cv2.putText(frame,'Count:'+str(score),(100,height-20), font, 1,(255,255,255),1)
            if(score>10):
                #person is feeling sleepy so we beep the alarm
                cv2.imwrite(os.path.join(path,'image.jpg'),frame)
                try:
                    sound.play()

                except:  # isplaying = False
                    pass
                if(thicc<16):
                    thicc= thicc+2
                else:
                    thicc=thicc-2
                    if(thicc<2):
                        thicc=2
                cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            #cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')











@app.route('/img_feed')
def img_feed():
    return Response(viola())


@app.route('/fe')
def fe():
    return render_template('fe.html')
@app.route("/fe_output.html", methods=["POST", "GET"])
def up_fe():
    UPLOAD_FOLDER = r"E:\Projects\Fyrproj\Project"
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    if request.method == "GET":
        return render_template("fe.html")

    else:
        file = request.files["file"]
        global upload_image_path
        upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        print(upload_image_path)
        file.save(upload_image_path)
        return render_template(
        "fe_output.html")
def viola():
    import numpy as np
    import cv2
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    face_cascade = cv2.CascadeClassifier(r'E:\Projects\ML\HaarCascadeFiles\haarcascade\haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(r'E:\Projects\ML\HaarCascadeFiles\haarcascade\haarcascade_eye.xml')
    print('Hi1')
    img = cv2.imread(upload_image_path)
    cv2.imshow('img',img)
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.resizeWindow('img', width, height)
    print('Hi2')
    faces = face_cascade.detectMultiScale(gray, 1.05, 10)
    for (x,y,w,h) in faces:
         cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
         roi_gray = gray[y:y+h, x:x+w]
         roi_color = img[y:y+h, x:x+w]
         eyes = eye_cascade.detectMultiScale(roi_gray)
         for (ex,ey,ew,eh) in eyes:
             cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    app.run()
