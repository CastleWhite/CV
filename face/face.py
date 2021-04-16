import cv2
import face_recognition

video_capture = cv2.VideoCapture("ted.mp4")
length = int(video_capture.get(7))
fps = video_capture.get(5)
size = (int(video_capture.get(3)),int(video_capture.get(4)))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_moive = cv2.VideoWriter('output_1.avi', fourcc, fps, size)

image = face_recognition.load_image_file("sample_face.png")
face_encoding = face_recognition.face_encodings(image)[0]
known_faces = [face_encoding,]

face_locations = []
face_encodings = []
face_names = []
frame_number = 0

while True:
    ret, frame = video_capture.read()
    frame_number += 1
    if not ret: 
        break

    rgb_frame = frame[:,:,::-1]

    face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    face_names = []
    for face_encoding in face_encodings:
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance = 0.5)
        name = None
        if match[0]:
            name = "the women"
        face_names.append(name)
    

    for (top, right, bottom, left),name in zip(face_locations,face_names):
        if not name: continue
        cv2.rectangle(frame, (left,top), (right,bottom), (0,0,255), 2)
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
    
    print("Writing frame {} / {}".format(frame_number, length))
    output_moive.write(frame)

video_capture.release()
cv2.destroyAllWindows()