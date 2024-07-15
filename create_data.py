import cv2
import os

# Path to the Haar Cascade XML file for face detection
haar_file = 'haarcascade_frontalface_default.xml'

# Folder where face images will be stored
datasets = 'datasets'

# Label for the sub-data set (your name)
sub_data = '#YOUR_NAME#'

# Create the datasets directory if it doesn't exist
if not os.path.exists(datasets):
    os.mkdir(datasets)

# Path to save face images
path = os.path.join(datasets, sub_data)
if not os.path.isdir(path):
    os.mkdir(path)

# Size of images to be saved
width, height = 130, 100

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + haar_file)

# Open the webcam
webcam = cv2.VideoCapture(0)

count = 1
while count <= 100:
    ret, im = webcam.read()
    if not ret:
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite(os.path.join(path, f'Pic_{count}.png'), face_resize)
        count += 1
        
    cv2.imshow('OpenCV', im)
    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()
