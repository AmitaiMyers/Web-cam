from datetime import datetime
import cv2, time,pandas
from datetime import datetime

#Create the data frame for start and end of seeing an object
df = pandas.DataFrame(columns=["Start","End"])
first_frame = None
status_list = [None,None]
times = []
video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()
    status = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if first_frame is None:
        first_frame = gray
        continue

# the frame and threshold
    delta_frame = cv2.absdiff(first_frame, gray)
    thresh_frame = cv2.threshold(delta_frame, 150, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    (cnts, _) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Whenever the cam detach an object
    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        status = 1
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    status_list.append(status)

    #Record the time change from 0->1 or 1->0
    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())

# Windows will appear
    cv2.imshow("Capturing", gray)
    cv2.imshow("Delta Frame", delta_frame)
    cv2.imshow("Threshold Frame", thresh_frame)
    cv2.imshow("Color Frame", frame)
    key = cv2.waitKey(1)

# Quite
    if key == ord('q'):
        if status == 1:
            times.append(datetime.now())
        break
    print(status)
print(status_list)
print(times)

# adding data to the csv
for i in range(0,len(times),2):
    df = df.append({"Start":times[i],"End":time[i+1]},ignore_index=True)
df.to_csv("Times.csv")
video.release()
cv2.destroyAllWindows


#Face recognition
# import cv2
# face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#
# img = cv2.imread("photo2.jpg")
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# faces = face_cascade.detectMultiScale(gray_img,
#                                       scaleFactor=1.05,minNeighbors=40)
# for x,y,w,h in faces:
#     img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3)
#
# resized = cv2.resize(img, (int(img.shape[1]/3), int(img.shape[0]/3)))
#
# cv2.imshow("Gray", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows

#Video
# import cv2, time
#
# video = cv2.VideoCapture(0)

#check is a boolean
#frame is a numpy array
# check,frame = video.read()
# # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# time.sleep(3)
# cv2.imshow("Capturing", frame)
#
#
# cv2.waitKey(0)
# video.release()
# cv2.destroyAllWindows