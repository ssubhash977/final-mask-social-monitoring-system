def social_dist():
	from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
	from tensorflow.keras.preprocessing.image import img_to_array
	from tensorflow.keras.models import load_model
	from imutils.video import VideoStream
	import numpy as np
	import argparse
	import imutils
	import time
	import cv2
	import os
	import streamlit as st
	from mylib import config, thread
	from mylib.mailer import Mailer
	from mylib.detection import detect_people
	from imutils.video import VideoStream, FPS
	from scipy.spatial import distance as dist
	import numpy as np
	import argparse, imutils, cv2, os, time, schedule

	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--face", type=str,
		default="face_detector",
			help="path to face detector model directory")
	ap.add_argument("-m", "--model", type=str,
			default="mask_detector.model",
			help="path to trained face mask detector model")
	ap.add_argument("-c", "--confidence", type=float, default=0.5,
			help="minimum probability to filter weak detections")
	args = vars(ap.parse_args())


	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", type=str, default="",
			help="path to (optional) input video file")
	ap.add_argument("-o", "--output", type=str, default="",
			help="path to (optional) output video file")
	ap.add_argument("-d", "--display", type=int, default=1,
			help="whether or not output frame should be displayed")
	args = vars(ap.parse_args())

		# load the COCO class labels our YOLO model was trained on
	labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
	LABELS = open(labelsPath).read().strip().split("\n")

	# derive the paths to the YOLO weights and model configuration
	weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
	configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

	# load our YOLO object detector trained on COCO dataset (80 classes)
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
	if config.USE_GPU:
		# set CUDA as the preferable backend and target
		print("")
		print("[INFO] Looking for GPU")
		net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
		net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

	# determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	# if a video path was not supplied, grab a reference to the camera
	if not args.get("input", False):
		print("[INFO] Starting the live stream..")
		vs = cv2.VideoCapture(config.url)
		if config.Thread:
				cap = thread.ThreadingClass(config.url)
		time.sleep(2.0)

	# otherwise, grab a reference to the video file
	else:
		print("[INFO] Starting the video..")
		vs = cv2.VideoCapture(args["input"])
		if config.Thread:
			cap = thread.ThreadingClass(args["input"])

	writer = None
	# start the FPS counter
	fps = FPS().start()


	st.title("Social Distancing Monitoring System")

	run2 = st.checkbox('Run Social Distancing model')
	st.header("Real time Social Distancing Monitoring")
	#col1,col2 = st.beta_rows(2)
	#col1.header('Real time Social Distancing Monitoring')
	#col2 = st.beta_columns(1)
	#col2.header('Video Input - Social Distancing Monitoring')

	FRAME_WINDOW = st.image([])
	
	FRAME_WINDOW2= st.image([])
	#FRAME_WINDOW2.header("Video - Input Social Distancing Monitoring")

	vs5 = cv2.VideoCapture("output.mp4")

	while run2:
		if config.Thread:
			frame = cap.read()

		else:
			(grabbed, frame) = vs.read()
			# if the frame was not grabbed, then we have reached the end of the stream
			if not grabbed:
				break
		frame = imutils.resize(frame, width=700)
		results = detect_people(frame, net, ln,
			personIdx=LABELS.index("person"))

		# initialize the set of indexes that violate the max/min social distance limits
		serious = set()
		abnormal = set()

		# ensure there are *at least* two people detections (required in
		# order to compute our pairwise distance maps)
		if len(results) >= 2:
			# extract all centroids from the results and compute the
			# Euclidean distances between all pairs of the centroids
			centroids = np.array([r[2] for r in results])
			D = dist.cdist(centroids, centroids, metric="euclidean")

			# loop over the upper triangular of the distance matrix
			for i in range(0, D.shape[0]):
				for j in range(i + 1, D.shape[1]):
					# check to see if the distance between any two
					# centroid pairs is less than the configured number of pixels
					if D[i, j] < config.MIN_DISTANCE:
						# update our violation set with the indexes of the centroid pairs
						serious.add(i)
						serious.add(j)
	                # update our abnormal set if the centroid distance is below max distance limit
					if (D[i, j] < config.MAX_DISTANCE) and not serious:
						abnormal.add(i)
						abnormal.add(j)

		# loop over the results
		for (i, (prob, bbox, centroid)) in enumerate(results):
			# extract the bounding box and centroid coordinates, then
			# initialize the color of the annotation
			(startX, startY, endX, endY) = bbox
			(cX, cY) = centroid
			color = (0, 255, 0)

			# if the index pair exists within the violation/abnormal sets, then update the color
			if i in serious:
				#color = (0, 0, 255)
				pass
			elif i in abnormal:
				color = (0, 0, 255) #orange = (0, 165, 255)

			# draw (1) a bounding box around the person and (2) the
			# centroid coordinates of the person,
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
			cv2.circle(frame, (cX, cY), 5, color, 2)

		# draw some of the parameters
		Safe_Distance = "Safe distance: >{} px".format(config.MAX_DISTANCE)
		cv2.putText(frame, Safe_Distance, (470, frame.shape[0] - 25),
			cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)
		#Threshold = "Threshold limit: {}".format(config.Threshold)
		#cv2.putText(frame, Threshold, (470, frame.shape[0] - 50),
			#cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)

	    # draw the total number of social distancing violations on the output frame
		# text = "Total serious violations: {}".format(len(serious))
		# cv2.putText(frame, text, (10, frame.shape[0] - 55),
		# 	cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)

		text1 = "Total Distancing violations: {}".format(len(abnormal))
		cv2.putText(frame, text1, (10, frame.shape[0] - 25),
			cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)

	#------------------------------Alert function----------------------------------#
		# if len(serious) >= config.Threshold:
		# 	cv2.putText(frame, "-ALERT: Violations over limit-", (10, frame.shape[0] - 80),
		# 		cv2.FONT_HERSHEY_COMPLEX, 0.60, (0, 0, 255), 2)
		# 	if config.ALERT:
		# 		print("")
		# 		print('[INFO] Sending mail...')
		# 		Mailer().send(config.MAIL)
		# 		print('[INFO] Mail sent')
		# 	#config.ALERT = False
	#------------------------------------------------------------------------------#
		# check to see if the output frame should be displayed to our screen
		
			# show the output frame
		#cv2.imshow("Real-Time Monitoring/Analysis Window", frame)

		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		
		FRAME_WINDOW.image(frame,caption='Real time Social Distancing Monitoring')
		ret, frame4 =  vs5.read()
		frame4 = cv2.cvtColor(frame4, cv2.COLOR_BGR2RGB)
		
		FRAME_WINDOW2.image(frame4,caption ='Video Input - Social Distancing Monitoring')
	st.header("Video - Input Social Distancing Monitoring")
