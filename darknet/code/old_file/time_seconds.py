import cv2
from PIL import Image 
import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))
sys.path.insert(0,'/home/zheng/Public/Graduation_Design/darknet/python')
import darknet as dn
import pdb
import time
if __name__ == "__main__":
    net = dn.load_net("/home/zheng/Public/Graduation_Design/darknet/cfg/yolov3.cfg".encode('utf-8'), "/home/zheng/Public/Graduation_Design/darknet/python/yolov3.weights".encode('utf-8'), 0)
    meta = dn.load_meta("/home/zheng/Public/Graduation_Design/darknet/cfg/coco.data".encode('utf-8'))
    statictmp="/home/zheng/Public/Graduation_Design/darknet/statictmp.jpg"
    dynamictmp="/home/zheng/Public/Graduation_Design/darknet/dynamictmp.jpg"
    # make a video_object and init the video object
    cap = cv2.VideoCapture("/home/zheng/Public/Graduation_Design/darknet/video/3.mp4")
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
            (h,w)=frame.shape[:2]
            h=h//2
            w=w//2
            print (h)
            print (w)
            frame = cv2.resize(frame,(w,h),fx=scaling_factor,fy=scaling_factor,interpolation=cv2.INTER_AREA) 
            start=time.time()
            #(h,w)=frame.shape[:2]     
            staticImg = frame[0:200,0:w]
            dynamicImg = frame[250:h,0:w]
            staticImg = Image.fromarray(staticImg)
            dynamicImg = Image.fromarray(dynamicImg)
            sg = staticImg.save(statictmp)
            dg = dynamicImg.save(dynamictmp)
            #cv2.imwrite(statictmp,staticImg)
            #cv2.imwrite(dynamictmp,dynamicImg)
            staticlabel = dn.detect(net, meta, statictmp.encode('utf-8'))
            dynamiclabel = dn.detect(net, meta, dynamictmp.encode('utf-8'))
            print (staticlabel)
            print (dynamiclabel)
            #print (staticlabel.count("car"))
            #print (dynamiclabel.count("person"))
            #print (staticlabel.count("car"))
            #print (dynamiclabel.count("person"))
            end=time.time()
            print(end-start)
            print ('#*********************************#')
            #display the rectangle of the objects in window
        else:
            #print("continue")
            continue
        # wait 1ms per iteration; press Esc to jump out the loop 
        c = cv2.waitKey(1)
        if (c==27) or (0xFF == ord('q')):
            break
    # release and close the display_window
    cap.release()
