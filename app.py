from __future__ import division, print_function
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from keras.models import load_model
import tensorflow as tf
import statistics as st
from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route('/capture', methods=['GET', 'POST'])
def camera():
    GR_dict = {0: (0, 255, 0), 1: (0, 0, 255)}


    model = tf.keras.models.load_model('final_model.h5')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    output = []
    cap = cv2.VideoCapture(0)

    i = 0
    while (i <= 30):
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.05, 5)

        for x, y, w, h in faces:

            face_img = img[y:y+h, x:x+w]

            resized = cv2.resize(face_img, (224, 224))
            reshaped = resized.reshape(1, 224, 224, 3)/255
            predictions = model.predict(reshaped)

            # find max indexed array
            max_index = np.argmax(predictions[0])

            emotions = ('angry', 'disgust', 'fear', 'happy',
                        'sad', 'neutral', 'surprise')
            predicted_emotion = emotions[max_index]
            output.append(predicted_emotion)

            cv2.rectangle(img, (x, y), (x+w, y+h), GR_dict[1], 2)
            cv2.rectangle(img, (x, y-40), (x+w, y), GR_dict[1], -1)
            cv2.putText(img, predicted_emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        i = i+1

        cv2.imshow('LIVE', img)
        key = cv2.waitKey(1)
        if key == 10:
            cap.release()
            cv2.destroyAllWindows()
            break
    print(output)
    cap.release()
    cv2.destroyAllWindows()

    final_output1 = st.mode(output)
    final_output1
    return render_template("capture.html",final_output=final_output1)

@app.route('/templates/capture_emotion', methods = ['GET','POST'])
def capture():
    return render_template("capture.html")

@app.route('/songs/angry', methods = ['GET', 'POST'])
def songsAngry():
    return render_template("songs_angry.html")

@app.route('/songs/fear', methods = ['GET', 'POST'])
def songsfear():
    return render_template("songs_fear.html")

@app.route('/songs/happy', methods = ['GET', 'POST'])
def songshappy():
    return render_template("songs_happy.html")

@app.route('/songs/neutral', methods = ['GET', 'POST'])
def songsneutral():
    return render_template("songs_neutral.html")

@app.route('/songs/sad', methods = ['GET', 'POST'])
def songssad():
    return render_template("songs_sad.html")

@app.route('/songs/surprise', methods = ['GET', 'POST'])
def songsSurprise():
    return render_template("songs_surprise.html")


if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0')

    