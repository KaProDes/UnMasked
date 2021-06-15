import cv2
import numpy as np
import os
from datetime import datetime
import sys

for p in sys.path:
    print(p)


def formatDate(dateTimeObj):
    dateTimeObjArray = str(dateTimeObj).split()
    dateComponents = dateTimeObjArray[0].split("-")
    trueDate = dateComponents[0]+"_"+dateComponents[1]+"_"+dateComponents[2]
    timeComponents = dateTimeObjArray[1].split(":")
    trueTime = timeComponents[0]+"_"+timeComponents[1]+"_"+timeComponents[2]
    fTimeStamp = trueDate + "_" + trueTime
    fTimeStamp = fTimeStamp.split(".")[0]
    return fTimeStamp


net = cv2.dnn.readNet('yolov3_training_final.weights', 'yolov3_testing.cfg')

classes = []
with open("classes.txt", "r") as f:
    classes = f.read().splitlines()

# cam_index specifies which camera is currently in use
# For troubleshooting use "video_name.mp4"
cam_index = 0
cap = cv2.VideoCapture(cam_index)
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

while True:
    _, img = cap.read()
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(
        img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + confidence,
                        (x, y+20), font, 2, (255, 255, 255), 2)

    cv2.imshow("Masked/Unmasked App", img)
    key = cv2.waitKey(1)
    # 27 is esc (to exit) and 32 is spacekey (to capture frame)
    if key == 27:
        print("Escape Pressed, Keyboard Interrupt. Exiting...")
        break
    elif key == 32:
        dateTimeObj = datetime.now()
        fTimeStamp = formatDate(dateTimeObj)
        img_name = "Captured_Image_{cam}_{timeStamp}.png".format(
            cam=cam_index, timeStamp=fTimeStamp)
        cv2.putText(img, img_name.split(".")[0], (10, 10), font, 1,
                    (100, 155, 255), 2, cv2.LINE_4)
        path_to_screenshots = "./screenshots"
        cv2.imwrite(os.path.join(path_to_screenshots, img_name), img)


cap.release()
cv2.destroyAllWindows()
