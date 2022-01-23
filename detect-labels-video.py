import boto3
import cv2
import base64

# Step by step
# 1. initialize rekogniotion client
rekog_client = boto3.client('rekognition')

# 2 load data inputnya (image atau video)
# img = cv2.imread('dog-and-cat-cover.jpg')
# _, im_arr = cv2.imencode('.jpg', img)  
# im_bytes = im_arr.tobytes()
# # im_b64 = base64.b64encode(im_bytes)
# print(type(im_bytes))

cap = cv2.VideoCapture('/dev/video1')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

while True:

    ret, frame = cap.read()
    H_ORI, W_ORI, channels = frame.shape
    ret, buf = cv2.imencode('.jpg', frame)
    
    # 3 inference using rekognition API
    response = rekog_client.detect_labels(
        Image={
            'Bytes': buf.tobytes(),
        },
        MaxLabels=123,
        MinConfidence=50
    )

    # print(response)

    # step 4: Draw or Visualize BB to original image
    
    for res in response['Labels']:
        bbox = res['Instances']
        if bbox != []:
            txt_label = res['Name']
            score = res['Confidence']
            txt_cv = f'{txt_label} - {score}%'
            # (x1,y1) = top left corner
            # (x2,y2) = bottom right corner
            x1 = int(bbox[0]['BoundingBox']['Left']*W_ORI)
            y1 = int(bbox[0]['BoundingBox']['Top']*H_ORI)
            x2 = int((bbox[0]['BoundingBox']['Left']+bbox[0]['BoundingBox']['Width'])*W_ORI)
            y2 = int((bbox[0]['BoundingBox']['Top']+bbox[0]['BoundingBox']['Height'])*H_ORI)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, txt_cv, (x1,y1), 1, 1, (0,0,255), 2)

    out.write(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# store img/video to S3 bucket

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
