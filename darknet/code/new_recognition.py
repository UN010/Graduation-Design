import time
import cv2
import numpy as np
def recognition_method(frame,ln,net,LABELS):
	(W, H) = (None, None)	
	number=np.zeros((3),dtype=int)
	if W is None or H is None:
		(H, W) = frame.shape[:2]
	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()
	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []
	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > 0.5:												#confidence
				#print(LABELS[classID])
				if LABELS[classID]=="car":
					number[0]=number[0]+1
				elif LABELS[classID]=="person":
					number[1]=number[1]+1
				else:
					continue
	print("car "+str(number[0]))
	print("person "+str(number[1]))
	number[2]=int((end-start+0.05)*60)
	print(end-start)
	return ((number,end-start))
