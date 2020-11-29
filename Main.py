# cv2 : This provide us functions to capture frames and
#       apply operations on them.
# pyttsx3 : It is used to generate voice incase of any movement.
# threading : It helps to generate frame and say any word at the
#             same time when there is any movement.

import cv2
import pyttsx3
from threading import *

engine = pyttsx3.init("sapi5")  #sapi5 is Microsoft Speech API
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[0].id) # Here we have many voices.
# We select 0 for a man voice.

# We create threads for saying Intruder in case of any movement.
# And this also help to say something while video is ongoing
# That is video and speak are running parallel.
class Speak(Thread):
    def run(self):
        engine.say("Intruder")
        engine.runAndWait()

# This helps to capture video frame of video : BurglaryVideo.xsec
# Note: BuglaryVideo.xesc is stored in our system
cap = cv2.VideoCapture("BurglaryVideo.xesc")

# To feed live camera then uncomment just succeding statement.
#cap = cv2.VideoCapture(0)

# cap.read() returns two things:
#   1. status: (True/False) Video frame is available to read or not.
#   2. A particular frame of video.
ret, frame1 = cap.read()
ret, frame2 = cap.read()

# If video can be opened or not i.e, video path is correct or not
while cap.isOpened():

    # Find difference btw two consecutive frames.
    diff = cv2.absdiff(frame1, frame2)

    # For doing better analysis we convert difference into gray scale.
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Gaussian Blur helps to remove High Frequency Noise.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold is used to classify the image.
    # Basically it highlights the details we want.
    # It assign 3rd arg to pixels which have pixel_value>=2nd arg
    _, th = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)

    # dilated helps to expand the threshold image.
    # Thin line becomes thick.
    dilated = cv2.dilate(th, None, iterations=3)

    # contours stores the list of boundary points.
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    for contour in contours:

        # (x, y) is a coordinate of a particular contour.
        (x, y, width, height) = cv2.boundingRect(contour)

        # If region of movement is big like for human and dog.
        # Avoids small false movement generate due to grains in video.
        if cv2.contourArea(contour) < 900:
            continue
        # Draw rectangle on moving object.
        cv2.rectangle(frame1, (x, y), (x+width, y+height), (0, 0, 255), 3)
        # It put text Intruder at top left corner in case of any movement.
        cv2.putText(frame1, "Intruder", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        # It generates a thread to say "Intruder".
        t1 = Speak()
        t1.start()
    cv2.imshow("Motion Detecting Frame", frame1)

    frame1 = frame2
    ret, frame2 = cap.read()

    # Quits when "q" is pressed.
    k = cv2.waitKey(1)
    if k & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()