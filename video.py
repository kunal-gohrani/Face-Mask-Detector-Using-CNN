import cv2
import maskDetectorScript
from skimage.exposure import adjust_gamma
mask = maskDetectorScript.MainScript()

v = cv2.VideoCapture(0)
v.open(0)
classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# flag=0
while True:
    ret,frame = v.read()
    resized=frame
    faces = classifier.detectMultiScale(resized, scaleFactor=1.06,
                                        minNeighbors=6,
                                        minSize=(20, 20))
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Extracting the faces
            face = resized[y:y + h, x :x + w]
            # Increasing the brightness of the extracted face, helps the model to classify. You can try changing the
            # gamma values to increase decrease brightness of image
            face = adjust_gamma(face,gamma=0.4,gain=1)
            # Sending the image to the model for classification
            text = mask.predict_image(cv2.cvtColor(face,cv2.COLOR_BGR2RGB))
            # Drawing a rectangle over the detected face
            cv2.rectangle(resized, (x-50, y-50), (x+w+50, y+h+50), (0, 255, 0), 2)
            # Debug output of the model in console
            print(text)

            font = cv2.FONT_HERSHEY_SIMPLEX
            # if flag==0:
            #     cv2.imwrite('kunal.jpg',face)
            #     flag=1

            # org
            org = (50, 50)

            # fontScale
            fontScale = 1

            # Blue color in BGR
            color = (255, 0, 0)

            # Line thickness of 2 px
            thickness = 2

            # Using cv2.putText() method
            resized = cv2.putText(resized, text, org, font,
                                fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow('image', resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
v.release()
cv2.destroyAllWindows()
