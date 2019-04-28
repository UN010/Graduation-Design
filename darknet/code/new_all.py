# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import sys
sys.path.append(os.path.join(os.getcwd(),'python/'))
sys.path.insert(0,'/home/zheng/Public/Graduation_Design/darknet/python/code')
import node_class
import new_recognition as rec
import light_change as cha
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
number=np.zeros((3),dtype=int)
list_instance1=node_class.CycleSingleLinkList()
list_instance2=node_class.CycleSingleLinkList()
light1=list_instance1._head
light2=list_instance1._head.next.next
top_time=90						#时间阈值(时间最大值)
change_time=300						#变换阈值(时间因子阈值)
temporary_time=0					#时间因子
while True:
	start=time.time()
	# read the next frame from the file
	(grabbed, frame) = vs.read()
	if number[2]!=0:
		number[2]=number[2]-1
		continue
	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break
	# if the frame dimensions are empty, grab them
	print(frame.shape)
	frame = frame[400:1080,0:1920]
	tup = rec.recognition_method(frame,ln,net,LABELS)
	number=tup[0]
	temporary_time=temporary_time+cha.change(tup)
	if temporary_time>=300 or time.time()-start>=90:
		light1=light1.next
		light2=light2.next
		temporary_time=0
	if str(light1)=='yellow' or str(light2)=='yellow':
		if temporary_time>=60:
			light1=light1.next
			light2=light2.next
			temporary_time=0
	print("红绿灯1"+str(light1))
	print("红绿灯2"+str(light2))
# release the file pointers
print("[INFO] cleaning up...")
vs.release()
