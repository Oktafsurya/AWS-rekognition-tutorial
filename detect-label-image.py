import boto3
import cv2
import base64

# Step by step
# 1. initialize rekogniotion client
rekog_client = boto3.client('rekognition')

# 2 load data inputnya (image atau video)
img = cv2.imread('dog-and-cat-cover.jpg')
_, im_arr = cv2.imencode('.jpg', img)  
im_bytes = im_arr.tobytes()
# im_b64 = base64.b64encode(im_bytes)
print(type(im_bytes))


# 3 inference using rekognition API
response = rekog_client.detect_labels(
    Image={
        'Bytes': im_bytes,
    },
    MaxLabels=5,
    MinConfidence=95
)

print(response)

# step 4: Draw or Visualize BB to original image
H_ORI, W_ORI, channels = img.shape
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
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img, txt_cv, (x1,y1), 1, 1, (0,0,255), 2)

# store img to S3 bucket

cv2.imshow('frame', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

