# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
sys.path.append(os.path.join(os.getcwd(),'python/'))
sys.path.insert(0,'/home/zheng/Public/Graduation_Design/darknet/python/code')
import node_class.py
labelsPath = "/home/zheng/Public/Graduation_Design/darknet/data/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")
# derive the paths to the YOLO weights and model configuration
#weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])					#yolov3.weights					
#configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])							
weightsPath = "/home/zheng/Public/Graduation_Design/darknet/python/yolov3.weights" 
configPath = "/home/zheng/Public/Graduation_Design/darknet/cfg/yolov3.cfg"
# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# initialize the video stream, pointer to output video file, and
# frame dimensions
#vs = cv2.VideoCapture(args["input"])
vs = cv2.VideoCapture("/home/zheng/Public/Graduation_Design/darknet/video/3.mp4")
writer = None
(W, H) = (None, None)
# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))
# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1
# loop over frames from the video file stream
tframe=0
number=np.zeros((2),dtype=int)
list_instance=node_class.CycleSingleLinkList()
light=list_instance._head
top_time=90
change_time=90
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()
	if tframe!=0:
		tframe=tframe-1
		continue
	# if the frame was not grabbed, then we have reached the end
	# of the stream
	number[0]=number[1]=0
	if not grabbed:
		break
	# if the frame dimensions are empty, grab them
	print(frame.shape)
	frame = frame[400:1080,0:1920]
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
	print(end-start)
	if change_time<=0:
		
	tframe=int((end-start+0.05)*60)
# release the file pointers
print("[INFO] cleaning up...")
vs.release()
