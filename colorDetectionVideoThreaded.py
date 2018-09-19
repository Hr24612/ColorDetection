# import the necessary packages
from __future__ import print_function
from imutils.video.pivideostream import PiVideoStream
import numpy as np
#import argparse
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
from imutils.video import FPS
import argparse
import imutils

# initialize the camera and grab a reference to the raw camera capture
#camera = PiCamera()
#camera.resolution = (640, 480)
#camera.framerate = 90
#rawCapture = PiRGBArray(camera, size=(640, 480))
 
vs=PiVideoStream()
vs.start()
time.sleep(2.0)
fps=FPS()
fps.start()
# allow the camera to warmup
#time.sleep(0.1)
# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", help = "path to the image")
#args = vars(ap.parse_args())
i=0
# load the image
#image = cv2.imread(args["image"])
#tick=time.time()
#for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
while True:

	i+=1
	image=vs.read()

	#image = frame.array
	#image=cv2.resize(image,(0,0),fx=.25,fy=.25)
	imageHsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
	#cv2.imshow('hsv',imageHsv)
#cv2.imshow("image",imageHsv)
# define the list of boundaries
#boundaries = [
#	([17, 15, 100], [50, 56, 200]),
#	([86, 31, 4], [220, 88, 50]),
#	([25, 146, 190], [62, 174, 250]),
#	([103, 86, 65], [145, 133, 128])
#]
#BGR Yellow (30,210,255)
#BGR Blue (144,64,14)
#BGR Pink (95,86,255)

	yellow=np.uint8([[[30,210,255]]])
	yellowHsv=cv2.cvtColor(yellow,cv2.COLOR_BGR2HSV)
	blue=np.uint8([[[144,64,14]]])
	blueHsv=cv2.cvtColor(blue,cv2.COLOR_BGR2HSV)
	pink=np.uint8([[[95,86,255]]])
	pinkHsv=cv2.cvtColor(pink,cv2.COLOR_BGR2HSV)
#print yellowHsv
#print blueHsv
#print pinkHsv
	boundariesHsv=[
		([24,20,20],[34,255,255]),#yellow
		([100,20,20],[160,255,255]),#blue
		([0,0,0],[5,255,255]),#pink
		([170,100,100],[180,255,255])#pink
	]
#boundaries=[
#	([10,190,235], [50,230,255]),
#	([124,44,0], [164,84,34]),
#	([75,66,235],[115,106,255])
#]
#size=(w,h,channels)
	height,width,channels=image.shape
	mask2=np.zeros([height,width],dtype=np.uint8)
#cv2.imshow("allBlack",mask2)
	# loop over the boundaries
	for (lower, upper) in boundariesHsv:
	# create NumPy arrays from the boundaries
		lower = np.array(lower, dtype = "uint8")
		upper = np.array(upper, dtype = "uint8")
	#lower=cv2.cvtColor(lower,cv2.COLOR_BGR2HSV)
	#upper=cv2.cvtColor(upper,cv2.COLOR_BGR2HSV)
	#boundaries=cv2.cvtColor(boundaries,cv2.COLOR_RGB2HSV)
	# find the colors within the specified boundaries and apply
	# the mask
		mask = cv2.inRange(imageHsv, lower, upper)
		mask2=mask2+mask
	#mask=cv2.bitwise_not(mask)
	output = cv2.bitwise_and(imageHsv, imageHsv, mask = mask2)
	blurred=cv2.GaussianBlur(mask2,(5,5),0)
	thresh=cv2.threshold(blurred,60,255,cv2.THRESH_BINARY)[1]
#cv2.imshow("thresh",thresh)
	cnts=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	cnts=cnts[1]
	cnts=sorted(cnts,key=cv2.contourArea,reverse=True)[:5]
#cnts=cnts[0,2]
#print cv2.contourArea(cnts[1])
	for c in cnts:
		if cv2.contourArea(c)>10000:		
			cv2.drawContours(image,[c],-1,(0,255,0),2)
	
#(cnts,_)=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#cnts=sorted(cnts,key=cv2.contourArea,reverse=True)[:5]
#for (i,c) in enumerate(cnts):
#	image=draw_contour(image,c,i)
#for cnt in contours:
#	print len(cnt)
#	cnt=np.squeeze(cnt)
#	cv2.drawContours(image,contours,cnt,(0,255,0),3)
	# show the images
	#cv2.imshow("images", np.hstack([image, output]))
	#cv2.imshow("images",image)
	
	#print i/(time.time()-tick)
	key = cv2.waitKey(1) & 0xFF
	#rawCapture.truncate(0)
	#print("[INFO] approx. FPS: {:.2f}".format(fps.update()))
	print (i)
	fps.update()
	if key == ord("q"):
		vs.stop()
fps.stop()
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
#tock=time.time()-tick
#framesPerSecond=i/tock
#print framesPerSecond
cv2.destroyAllWindows()
vs.stop()


