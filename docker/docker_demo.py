from pydarknet import Detector, Image
import cv2
import os
import time
import boto3
s3 = boto3.resource('s3',aws_access_key_id='AKIAVJRGT2S4WXW4SYVB', aws_secret_access_key='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')

bucket = s3.Bucket('yolo-input')

if __name__ == "__main__":
    net = Detector(bytes("cfg/yolov3-data.cfg", encoding="utf-8"), bytes("weights/yolov3-data_final_26.weights", encoding="utf-8"), 0, bytes("cfg/voc.data",encoding="utf-8"))

    # input_files = os.listdir("s3_input")
    for file in bucket.objects.all():
    # for file_name in input_files:
        key = file.key.encode('utf-8')
        # bucket.download_file(key, f"../s3_input/{key}")
        file_name = key
        if not file_name.lower().endswith(".jpg"):
            continue

        print("File:", key)
        img = cv2.imread(os.path.join("../s3_input",key))
        img2 = Image(img)

        start_time = time.time()
        results = net.detect(img2)
        end_time = time.time()

        print("Elapsed Time:", end_time-start_time)

        for cat, score, bounds in results:
            x, y, w, h = bounds
            cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), thickness=2)
            cv2.putText(img,str(cat.decode("utf-8")),(int(x),int(y)),cv2.FONT_HERSHEY_DUPLEX,4,(0,0,255), thickness=1)


        cv2.imwrite(os.path.join("../s3_output",file_name), img)
        # s3.meta.client.upload_file(f"../s3_output/{key}", 'yolo-output', key )
        os.remove(os.path.join("../s3_output",file_name))
