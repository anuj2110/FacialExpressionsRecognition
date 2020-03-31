import cv2 
from tensorflow.keras.models import load_model
import numpy as np

cascade_file = "haarcascade_frontalface_default.xml"
model_file = "fermodel.h5"
emotions = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]

face_detection = cv2.CascadeClassifier(cascade_file)
emotion_classifier = load_model(model_file, compile=False)
emotion_=0
# img = cv2.imread("images.jpg")
# grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# faces = face_detection.detectMultiScale(grayimg, 1.3, 5)
# print(faces)
# for (x,y,w,h) in faces:
#     detected_face = grayimg[int(y):int(y+h),int(x):int(x+w)]
#     detected_face = cv2.resize(detected_face,(48,48),interpolation = cv2.INTER_AREA)
#     detected_face = np.array(detected_face).reshape((1,48,48,1))/255.0
#     label = np.argmax(emotion_classifier.predict(detected_face))
#     emotion_=label
#     emotion = emotions[label]
#     print(emotion)
#     cv2.putText(img, emotion, (int(x+w/2),y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255,0), 2)
#     cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
# cv2.imshow("",img)
# cv2.waitKey(0)
cap = cv2.VideoCapture(0)
#capture webcam

while(True):
    ret, img = cap.read()
    faces = face_detection.detectMultiScale(img, 1.3, 5)
    for (x,y,w,h) in faces:
        detected_face = img[int(y):int(y+h),int(x):int(x+w)]
        detected_face = cv2.cvtColor(detected_face,cv2.COLOR_BGR2GRAY)
        detected_face = cv2.resize(detected_face,(48,48),interpolation = cv2.INTER_AREA)
        detected_face = np.array(detected_face).reshape((1,48,48,1))/255.0
        label = np.argmax(emotion_classifier.predict(detected_face))
        emotion_=label
        emotion = emotions[label]
        cv2.putText(img, emotion, (x+w,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255,0), 2)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
    cv2.imshow("",img)
    print(emotion_)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

        # When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()