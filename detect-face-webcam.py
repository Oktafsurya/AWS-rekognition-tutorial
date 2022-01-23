import cv2
import numpy as np
import boto3

# Setup
scale_factor = .15
green = (0,255,0)
red = (0,0,255)
frame_thickness = 2
cap = cv2.VideoCapture('/dev/video1')
rekognition = boto3.client('rekognition')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

while(True):

    # Capture frame-by-frame
    ret, frame = cap.read()
    height, width, channels = frame.shape

    # Convert frame to jpg
    small = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)))
    ret, buf = cv2.imencode('.jpg', small)

    # Detect faces in jpg
    faces = rekognition.detect_faces(Image={'Bytes':buf.tobytes()}, Attributes=['ALL'])

    print(faces)
    print(f'='*60)

    # Draw rectangle around faces
    for face in faces['FaceDetails']:
        smile = face['Smile']['Value']
        gender_label = face['Gender']['Value']
        gender_score = face['Gender']['Confidence']
        age_low = face['AgeRange']['Low']
        age_high = face['AgeRange']['High']
        emotion_type = face['Emotions'][0]['Type']
        emotion_score = face['Emotions'][0]['Confidence']
        text_gender = f'{gender_label} - {gender_score}%'
        text_age = f'interval age: {age_low} - {age_high}'
        text_emotion = f'Emotion: {emotion_type} - {emotion_score}%'
        x1 = int(face['BoundingBox']['Left']*width)
        y1 =  int(face['BoundingBox']['Top']*height)
        x2 = int((face['BoundingBox']['Left']+face['BoundingBox']['Width'])*width)
        y2 = int((face['BoundingBox']['Top']+face['BoundingBox']['Height'])*height)
        cv2.rectangle(frame, (x1,y1), (x2,y2), green if smile else red, frame_thickness)
        cv2.putText(frame, text_gender, (x1,y1), 1, 1, (0,0,255), 2)
        cv2.putText(frame, text_age, (x1,y1-30), 1, 1, (0,0,255), 2)
        cv2.putText(frame, text_emotion, (x1,y2), 1, 1, (0,0,255), 2)

    # Display the resulting frame
    out.write(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()