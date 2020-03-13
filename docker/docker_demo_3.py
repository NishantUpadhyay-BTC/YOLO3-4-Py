import cv2
import numpy as np
# import pytesseract
import boto3
s3 = boto3.resource('s3',aws_access_key_id=os.getenv("ACCESS_KEY"), aws_secret_access_key=os.getenv("ACCESS_SECRET"))
bucket = s3.Bucket('yolo-input')

file = list(bucket.objects.all())[0]
key = file.key
bucket.download_file(key, 'input/{0}'.format(key))

net = cv2.dnn.readNetFromDarknet("weights/yolov3-data_final_26.weights","cfg/yolov3-data.cfg")
print(net)

classes = ["shipper","consignee","email"]
layer_names = net.getLayerNames()

outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors= np.random.uniform(0,255,size=(len(classes),3))


img = cv2.imread(os.path.join("input",key))
# img = cv2.resize(img,None,fx=0.4,fy=0.3)
height,width,channels = img.shape

blob = cv2.dnn.blobFromImage(img,0.00392,(416,416),(0,0,0),True,crop=False)

net.setInput(blob)
outs = net.forward(outputlayers)
#print(outs[1])


#Showing info on screen/ get confidence score of algorithm in detecting an object in blob
class_ids=[]
confidences=[]
boxes=[]
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            #onject detected
            center_x= int(detection[0]*width)
            center_y= int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)

            #cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
            #rectangle co-ordinaters
            x=int(center_x - w/2)
            y=int(center_y - h/2)
            #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

            boxes.append([x,y,w,h]) #put all rectangle areas
            confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
            class_ids.append(class_id) #name of the object tha was detected

indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)


font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    print(i)
    if i in indexes:
        x,y,w,h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        cv2.putText(img,label,(x,y+30),font,1,(255,255,255),2)
        crop_image = img[y:y+h,x:x+w]
        # value = pytesseract.image_to_string(crop_image,lang="chi+chi_tra+eng", config=("--psm 6"))
        # print(value)
cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
