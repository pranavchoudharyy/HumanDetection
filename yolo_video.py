import numpy as np
import argparse
import imutils
import time
import cv2
import os
ap = argparse.ArgumentParser()
ap.add_argument("-i", required=True,
	help="Denotes the source image/video path")
ap.add_argument("-y", required=True,
	help="To provide the path to Yolo COCO model")
ap.add_argument("-c", type=float, default=0.5,
	help="Minimum value to discard weak detection")
ap.add_argument("-t", type=float, default=0.3,
	help="Threshold value for non max suppression ")
args = vars(ap.parse_args())
labelsPath = os.path.sep.join([args["y"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")
weightsPath = os.path.sep.join([args["y"], "yolov3.weights"])
configPath = os.path.sep.join([args["y"], "yolov3.cfg"])
print("loading...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
vs = cv2.VideoCapture(args["i"])
writer = None
(W, H) = (None, None)
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("Total frames in video : {} ".format(total))
except:
	print("Error: E/could not determine # of frames in video")
	print("Error: E/no approx. completion time can be provided")
	total = -1
while vs.isOpened():
	(grabbed, frame) = vs.read()
	if not grabbed:
		break
	if W is None or H is None:
		(H, W) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()
	boxes = []
	confidences = []
	classIDs = []
	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			if confidence > args["c"]:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["c"],
		args["t"])
	if len(idxs) > 0:
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]],
				confidences[i])
			cv2.putText(frame, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			cv2.imshow("people", frame)
			key = cv2.waitKey(1)
print("Finishing...")
