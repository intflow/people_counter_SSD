# USAGE
# To read and write back out to video:
# python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/example_01.mp4 \
#	--output output/output_01.avi
#
# To read from webcam and write back out to disk:
# python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel \
#	--output output/webcam_output.avi

# import the necessary packages
import numpy as np
import argparse
import imutils
from imutils.video import WebcamVideoStream
import time
import dlib
import cv2
from tracklib.centroidtracker import CentroidTracker
from tracklib.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import serial
import base64
import cv2
#import zmq
import sys
from pytz import timezone
from datetime import datetime
#from threading import Thread
from multiprocessing import Process, Queue
from time import sleep

#number formatter
def num_digit_str(num, digit):
	thr = 10 ** digit

	if num >= thr:
		num = thr - 1

	return str(num).zfill(digit)

# rescaled frame <=> axies synchronization
def rconv(x, y, r_x, r_y):
	x_hat = int(x * r_x + 0.5)
	y_hat = int(y * r_y + 0.5)
	return x_hat, y_hat

# Set functions for YOLO model
def get_output_layers(net):
	layer_names = net.getLayerNames()
	output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	return output_layers

def obj_detect_dnn(net, frame):
	
	# grab the frame from the input queue, resize it, and
	# construct a blob from it
	(H, W) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
	net.setInput(blob)
	detections = net.forward()
		
	# write the detections to the output queue
	return detections

def init_ipcam():
		print('----Reconnect RTSP IPCAM----')
		try:
			vs = cv2.VideoCapture("rtsp://admin:airwolf1!@192.168.1.64:554/Streaming/Channels/101/ch1-s1?tcp")
			if not vs.isOpened():
				print("----CAM1 load failed... Exiting----")
				sys.exit()
		except Warning:
			sys.exit()

		try:
			vs2 = cv2.VideoCapture("rtsp://admin:airwolf1!@192.168.1.65:554/Streaming/Channels/101/ch1-s1?tcp")
			if not vs2.isOpened():
				TWOCAM = 0
				print("----CAM2 load failed... Exiting----")
				sys.exit()
			else:
				TWOCAM = 1
		except Warning:
			sys.exit()	

		print("----CAM Number = " + str(TWOCAM+1) + "----")
		
		return vs, vs2, TWOCAM

def thread_vid(args, vs, outQue1, outQue2):
	while True:
		ret, frame = vs.read()
		if not(ret):
			vs = cv2.VideoCapture(args["input"])
			continue
		#Cut display frame by half
		(H, W) = frame.shape[:2]
		W_cut = int(W*0.15)
		frame = frame[:,W_cut:-W_cut,:]
		
		# resize the frame to have a minimum width(or height) supported by a model, then convert
		# the frame from BGR to RGB for dlib
		(H_org, W_org) = frame.shape[:2]
		W = 320
		H = 320
		frame = cv2.resize(frame, dsize=(320,320), interpolation=cv2.INTER_AREA)
		(H, W) = frame.shape[:2]
		r_x = W_org / W
		r_y = H_org / H		

		outQue1.put(ret)
		outQue2.put(frame)


		# show the output frame
		if args["screen"] == 1:
			# check to see if we should write the frame to disk
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF
			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break

		

def main(args):

	try:
		# Set UART initialization
		ser = serial.Serial(args["uart_port"], args["uart_baud"], timeout=1, parity='N', stopbits=1, bytesize=8)
		
		# open the serial port
		if ser.isOpen():
			print(ser.name + ' is open...')
			UART_WORK = 1
			ser.write(bytes('---AIcam Start---\r\n', encoding='ascii'))
	except:
		UART_WORK = 0

	# initialize the list of class labels MobileNet SSD was trained to
	# detect
	#txt_file = open("models/yolov3.txt", "r")
	#CLASSES = txt_file.read().split('\n')
	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
		
	#Color map for detection box
	COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
	COLORS[15] = [0, 0, 255] #person
	#COLORS[1] = [102, 255, 255] #bicycle
	#COLORS[2] = [255, 51, 0] #car
	#COLORS[3] = [102, 255, 255] #motorbike
	#COLORS[5] = [255, 102, 0] #bus
	#COLORS[6] = [160, 102, 100] #train
	#COLORS[7] = [185, 152, 250] #truck

	# initialize the video writer (we'll instantiate later if need be)
	#writer = None	

	print("[INFO] starting nnet obj detection...")
	print("[INFO] loading model...")
	net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
	
	# instantiate our centroid tracker, then initialize a list to store
	# each of our dlib correlation trackers, followed by a dictionary to
	# map each unique object ID to a TrackableObject
	ct = CentroidTracker(maxDisappeared=5, maxDistance=20)


	tracker = cv2.TrackerMOSSE_create
	trackers = []
	trackableObjects = {}

	# initialize the total number of frames processed thus far, along
	# with the total number of objects that have moved either up or down
	totalFrames = 0
	totalDown = 0
	totalUp = 0

	### Initialize tcp socket for video streaming
	##context = zmq.Context()
	##footage_socket = context.socket(zmq.PUB)
	##footage_socket.connect(args["stream_ip"])

	# start the frames per second throughput estimator
	fps = FPS().start()

	#For rtsp streaming
	if sys.platform == "win32":
		import os, msvcrt
		msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)


	# Video Get Thread
	# if a video path was not supplied, grab a reference to the webcam
	if not args.get("input", False):
		print("[INFO] starting video stream...")
		
		[vs, vs2, TWOCAM] = init_ipcam()
		
		
	# otherwise, grab a reference to the video file
	else:
		print("[INFO] opening video file...")
		vs = cv2.VideoCapture(args["input"])
		TWOCAM = 0	

	outQue1 = Queue(maxsize=1)
	outQue2 = Queue(maxsize=1)
	p = Process(target=thread_vid, args=(args, vs, outQue1, outQue2))
	p.daemon = True
	p.start()
	p.join()

	# loop over frames from the video stream
	while True:
		ret = outQue1.get()
		if ret:
			frame = outQue2.get()
			# if we are viewing a video and we did not grab a frame then we
			# have reached the end of the video
			if args["input"] is not None and frame is None:
				break
			
			(H, W) = frame.shape[:2]
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			# initialize the current status along with our list of bounding
			# box rectangles returned by either (1) our object detector or
			# (2) the correlation trackers
			status = "Waiting"
			rects = [] #purge rects buffer
			classes = [] 
			#update current time
			time_fmt = "%Y-%m-%d-%H:%M"
			time_now = datetime.now(timezone('Asia/Seoul')).strftime(time_fmt)
			if datetime.now(timezone('Asia/Seoul')).strftime('%H:%M') == '23:59':
				totalDown_p = 0
				totalUp_p = 0
				totalDown_v = 0
				totalUp_v = 0
				totalUp_m = 0
			# check to see if we should run a more computationally expensive object detection method to aid our tracker
			if totalFrames % args["skip_frames"] == 0:
				# set the status and initialize our new set of object trackers
				status = "Detecting"
				trackers = []
				idx_stack = []
				# convert the frame to a blob and pass the blob through the
				# network and obtain the detections
				blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
				net.setInput(blob)
				detections = net.forward()
				# loop over the detections
				for i in np.arange(0, detections.shape[2]):
					# extract the confidence (i.e., probability) associated
					# with the prediction
					confidence = detections[0, 0, i, 2]
					# filter out weak detections by requiring a minimum
					# confidence
					if confidence > args["confidence"]:
						# extract the index of the class label from the
						# detections list
						idx = int(detections[0, 0, i, 1])
						# if the class label is not a person, ignore it
						if not(CLASSES[idx] == "person"):
							continue
						# compute the (x, y)-coordinates of the bounding box
						# for the object
						box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
						(startX, startY, endX, endY) = box.astype('long')  
						
						# construct a dlib rectangle object from the bounding
						# box coordinates and then start the dlib correlation
						# tracker
						tracker = dlib.correlation_tracker()# grab the new bounding box coordinates of the object
					
						rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
						##(success, box) = tracker.update(frame)
						##(x, y, w, h) = [int(v) for v in box]
						##cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
						tracker.start_track(rgb, rect)
						# add the tracker to our list of trackers so we can
						# utilize it during skip frames
						trackers.append(tracker)
						idx_stack.append(idx)
						
			# otherwise, we should utilize our object *trackers* rather than
			# object *detectors* to obtain a higher frame processing throughput
			else:
				idx_cnt = 0
				# loop over the trackers
				for tracker in trackers:
					# set the status of our system to be 'tracking' rather
					# than 'waiting' or 'detecting'
					status = "Tracking"
					# update the tracker and grab the updated position
					tracker.update(rgb)
					pos = tracker.get_position()
					# unpack the position object
					startX = int(pos.left())
					startY = int(pos.top())
					endX = int(pos.right())
					endY = int(pos.bottom())
					# add the bounding box coordinates to the rectangles list
					rects.append((startX, startY, endX, endY))
					
					# draw the prediction on the frame
					idx_target = idx_stack[idx_cnt]
					if (CLASSES[idx_target] == "person"):
						label = "{}: {:.2f}%".format(CLASSES[idx_target],
							confidence * 100)
						cv2.rectangle(frame, (startX, startY), (endX, endY),
							COLORS[idx_target], 2)
						y = startY - 15 if startY - 15 > 15 else startY + 15
						cv2.putText(frame, label, (startX, y),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx_target], 2)   
					
					idx_cnt = idx_cnt + 1
			# draw a horizontal line in the center of the frame -- once an
			# object crosses this line we will determine whether they were
			# moving 'up' or 'down'
			#화각의 틸트 상태에 따라 튜닝 필요 (기울어질수록 UP_LINE을 DOWN_LINE 쪽으로 당긴다.
			#DOWN_LINE = int(float(H) // 2.0)#1.2)
			#UP_LINE = int(float(H) // 2.0)#7.6)
			#cv2.line(frame, (0, DOWN_LINE), (W, DOWN_LINE), (255, 0, 0), 2)
			#cv2.line(frame, (0, UP_LINE), (W, UP_LINE), (0, 0, 255), 2)
			# use the centroid tracker to associate the (1) old object
			# centroids with (2) the newly computed object centroids
			objects = ct.update(rects)
			# loop over the tracked objects
			for (objectID, centroid) in objects.items():
				# check to see if a trackable object exists for the current
				# object ID
				to = trackableObjects.get(objectID, None)
				# if there is no existing trackable object, create one
				if to is None:
					to = TrackableObject(objectID, centroid)
				# otherwise, there is a trackable object so we can utilize it
				# to determine direction
				else:
					# the difference between the y-coordinate of the *current*
					# centroid and the mean of *previous* centroids will tell
					# us in which direction the object is moving (negative for
					# 'up' and positive for 'down')
					y = [c[1] for c in to.centroids]
					direction = centroid[1] - np.mean(y)
					to.centroids.append(centroid)
					# check to see if the object has been counted or not
					if not to.counted:
						to.counted = True
						# if the direction is negative (indicating the object
						# is moving up) AND the centroid is above the center
						# line, count the object
						#if direction < 0 and centroid[1] < UP_LINE:
						#	totalUp += 1
						#	to.counted = True

						## if the direction is positive (indicating the object
						## is moving down) AND the centroid is below the
						## center line, count the object
						#elif direction > 0 and centroid[1] > DOWN_LINE:
						#	totalDown += 1
						#	to.counted = True
				# store the trackable object in our dictionary
				trackableObjects[objectID] = to
				# draw both the ID of the object and the centroid of the
				# object on the output frame
				text = "ID {}".format(objectID)
				cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
				cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
			# construct a tuple of information we will be displaying on the
			# frame
			num_person = len(objects.items())
			info = [
				("Num", num_person),
				#("Status", status),
			]
			# loop over the info tuples and draw them on our frame
			for (i, (k, v)) in enumerate(info):
				text = "{}: {}".format(k, v)
				cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

			# increment the total number of frames processed thus far and
			# then update the FPS counter
			totalFrames += 1				
			fps.update()
			if totalFrames == args["update_ratio"]:
				totalFrames = 0
				print(num_person)
				if UART_WORK == 1:
					ser.write(num_person.to_bytes(1, byteorder='little'))
					print(num_person.to_bytes(1, byteorder='little'))


	# stop the timer and display FPS information
	#fps.stop()
	#print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	#print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	## check to see if we need to release the video writer pointer
	#if writer is not None:
	#	writer.release()

	# if we are not using a video file, stop the camera video stream
	if not args.get("input", False):
		vs.stop()

	# otherwise, release the video file pointer
	else:
		vs.release()

	# close any open windows
	cv2.destroyAllWindows()


if __name__ == '__main__':
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--prototxt", type=str, default="models/MobileNetSSD_deploy.prototxt",
		help="path to Caffe 'deploy' prototxt file")
	ap.add_argument("-m", "--model", type=str, default="models/MobileNetSSD_deploy.caffemodel",
		help="path to Caffe pre-trained model")
	ap.add_argument("-i", "--input", type=str, default="example_01.mp4",
		help="path to optional input video file")
	ap.add_argument("-o", "--output", type=str, default="",
		help="path to optional output video file")
	ap.add_argument("-c", "--confidence", type=float, default=0.3,
		help="minimum probability to filter weak detections")
	ap.add_argument("-s", "--skip-frames", type=int, default=5,
		help="# of skip frames between detections")
	ap.add_argument("-rf", "--resize-frame", type=int, default=320,
		help="minimum video frame's size for a given model")
	ap.add_argument("-od", "--object-diff", type=int, default=0.5,
		help="count threshold according to object's position difference")
	ap.add_argument("-lu", "--line-up", type=float, default=1,
		help="upper line control factor")
	ap.add_argument("-ld", "--line-down", type=float, default=1,
		help="down line control factor")
	ap.add_argument("-d", "--direction-count", type=int, default=2,
		help="0: Up/Down, 1: Right/Left")
	ap.add_argument("-ip", "--stream_ip", type=str, default="tcp://192.168.0.38:5555",
		help="IP for stream to video server")
	ap.add_argument("-up", "--uart_port", type=str, default="/dev/ttyS0",
		help="port number for UART communication")
	ap.add_argument("-ub", "--uart_baud", type=int, default=9600,
		help="baud rate for UART communication")
	ap.add_argument("-scr", "--screen", type=int, default=1,
		help="Turn On/Off OSD")
	ap.add_argument("-ur", "--update_ratio", type=int, default=30,
		help="Send ratio")

	
	args = vars(ap.parse_args())
	main(args)
