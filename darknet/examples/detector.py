# Stupid python path shit.
# Instead just add darknet.py to somewhere in your python path
# OK actually that might not be a great idea, idk, work in progress
# Use at your own risk. or don't, i don't care

import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))
sys.path.insert(0,'/home/zheng/Public/Graduation_Design/darknet/python')
import darknet as dn
import pdb
dn.set_gpu(0)
net = dn.load_net("/home/zheng/Public/Graduation_Design/darknet/cfg/yolov3-tiny.cfg".encode('utf-8'), "/home/zheng/Public/Graduation_Design/darknet/python/yolov3-tiny.weights".encode('utf-8'), 0)
meta = dn.load_meta("/home/zheng/Public/Graduation_Design/darknet/cfg/coco.data".encode('utf-8'))

# And then down here you could detect a lot more images like:
r = dn.detect(net, meta, "/home/zheng/Public/Graduation_Design/darknet/data/eagle.jpg".encode('utf-8'))
print (r)
r = dn.detect(net, meta, "/home/zheng/Public/Graduation_Design/darknet/data/giraffe.jpg".encode('utf-8'))
print (r)
r = dn.detect(net, meta, "/home/zheng/Public/Graduation_Design/darknet/data/horses.jpg".encode('utf-8'))
print (r)
r = dn.detect(net, meta, "/home/zheng/Public/Graduation_Design/darknet/data/person.jpg".encode('utf-8'))
print (r)

