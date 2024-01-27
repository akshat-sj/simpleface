import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import pickle
from multiprocessing import Pool


def findEncodings(img):
    #conver image back to rgb
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #get encoded face
    encoded_face = face_recognition.face_encodings(img, num_jitters=1)[0]
    return encoded_face

def markAttendance(name):
    with open('attendance.csv', 'r+') as f:
        data = f.readlines()
        nl = [entry.split(',')[0] for entry in data]
        if name not in nl:
            now = datetime.now()
            time = now.strftime('%I:%M:%S:%p')
            date = now.strftime('%d-%B-%Y')
            f.writelines(f'\n{name}, {time}, {date}')

path = 'img'
images = []
classNames = []
mylist = os.listdir(path)

for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# Resize images for faster face recognition
resized_images = [cv2.resize(img, (0, 0), fx=0.25, fy=0.25) for img in images]

if __name__ == '__main__':
    # Use multiprocessing to parallelize face encoding
    with Pool() as pool:
        encoded_face_train = pool.map(findEncodings, resized_images)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    frame_counter = 0
    skip_frames = 5  # Process every 5th frame

    while True:
        status,img = cap.read()
        #skip frame checker
        if frame_counter % skip_frames == 0:
            imgn = cv2.resize(img, (0, 0), None, fx=0.25, fy=0.25)
            imgn = cv2.cvtColor(imgn, cv2.COLOR_BGR2RGB)
            faces_loc = face_recognition.face_locations(imgn)
            encoded_faces = face_recognition.face_encodings(imgn, faces_loc, num_jitters=1)

            for encode_face, faceloc in zip(encoded_faces, faces_loc):
                matches = face_recognition.compare_faces(encoded_face_train, encode_face, tolerance=0.6)
                matchIndex = np.argmin(face_recognition.face_distance(encoded_face_train, encode_face))

                if matches[matchIndex]:
                    name = classNames[matchIndex].upper().lower()
                    y1, x2, y2, x1 = faceloc
                    # since we scaled down by 4 times
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    rectangle_color = (128, 0, 128)  # Purple color
                    rectangle_thickness = 2

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    font_thickness = 2
                    font_color = (255, 255, 255)  # White color

                    cv2.rectangle(img, (x1, y1), (x2, y2), rectangle_color, rectangle_thickness)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), rectangle_color, cv2.FILLED)
                    cv2.putText(img, name.capitalize(), (x1 + 6, y2 - 10), font, font_scale, font_color, font_thickness)
                    markAttendance(name)

            cv2.imshow('webcam', img)

        frame_counter += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()