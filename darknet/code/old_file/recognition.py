import cv2
from PIL import Image 
import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))
sys.path.insert(0,'/home/zheng/Public/Graduation_Design/darknet/python/code')
import darknet as dn
import pdb
import time
if __name__ == "__main__":
    net = dn.load_net("/home/zheng/Public/Graduation_Design/darknet/cfg/yolov3.cfg".encode('utf-8'), "/home/zheng/Public/Graduation_Design/darknet/python/yolov3.weights".encode('utf-8'), 0)
    meta = dn.load_meta("/home/zheng/Public/Graduation_Design/darknet/cfg/coco.data".encode('utf-8'))
    statictmp="/home/zheng/Public/Graduation_Design/darknet/statictmp.jpg"
    dynamictmp="/home/zheng/Public/Graduation_Design/darknet/dynamictmp.jpg"
    # make a video_object and init the video object
    cap = cv2.VideoCapture("/home/zheng/Public/Graduation_Design/darknet/3.mp4")
    # define picture to_down' coefficient of ratio
    scaling_factor = 0.5
    count = 0
    # loop until press 'esc' or 'q'
    while (cap.isOpened()):
        # collect current frame
        ret, frame = cap.read()
        if ret == True:
            count = count + 1
            #print count
        else:
            break
        #detect and show per 50 frames
        if count == 50:
            count = 0
            # resize the frame
            #frame = cv2.resize(frame,None,fx=scaling_factor,fy=scaling_factor,interpolation=cv2.INTER_AREA) 
            (h,w)=frame.shape[:2]     
            staticImg = frame[0:400,0:w]
            dynamicImg = frame[500:h,0:w]
            staticImg = Image.fromarray(staticImg)
            dynamicImg = Image.fromarray(dynamicImg)
            sg = staticImg.save(statictmp)
            dg = dynamicImg.save(dynamictmp)
            #img_arr = Image.fromarray(frame)
            #img_arr.save(video_tmp)
            staticlabel = dn.detect(net, meta, statictmp.encode('utf-8'))
            dynamiclabel = dn.detect(net, meta, dynamictmp.encode('utf-8'))
            print (staticlabel)
            print (dynamiclabel)
            print ('')
            print ('#*********************************#')
            #display the rectangle of the objects in window
        else:
            print("continue")
            continue
        # wait 1ms per iteration; press Esc to jump out the loop 
        c = cv2.waitKey(1)
        if (c==27) or (0xFF == ord('q')):
            break
    # release and close the display_window
    cap.release()
