

#----------------------------Parse req. arguments------------------------------#

def all_functions():
	from mylib import config, thread
	from mylib.mailer import Mailer
	from mylib.detection import detect_people
	from imutils.video import VideoStream, FPS
	from scipy.spatial import distance as dist
	import numpy as np
	import argparse, imutils, cv2, os, time, schedule



		# construct the argument parser and parse the arguments
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
	
	#------------------------------------------------------------------------------#

	# load the COCO class labels our YOLO model was trained on
	labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
	LABELS = open(labelsPath).read().strip().split("\n")

	# derive the paths to the YOLO weights and model configuration
	weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
	configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

	# load our YOLO object detector trained on COCO dataset (80 classes)
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

	# check if we are going to use GPU
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




	########################################################################################################


	import cv2
	import streamlit as st
	
	st.title("Social Distancing and Mask Detection Monitoring System")
	run = st.checkbox('Start')

	import both



	st.header("Social Distancing and Mask Detection Monitoring System")
	FRAME_WINDOW3= st.image([])
	col1, col2 = st.beta_columns(2)
	col1.header('Real time Face Mask Detection')
	col2.header('Real time Social Distancing Monitoring')
	st.sidebar.header('Bounding Box Conditions')
	st.sidebar.image('conditions.jpeg')
	FRAME_WINDOW = col1.image([])
	FRAME_WINDOW2= col2.image([])




	#camera = cv2.VideoCapture(0)

	#while run:
	   # _, frame = camera.read()
	   # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	#     FRAME_WINDOW.image(frame)
	#     FRAME_WINDOW2.image(frame)
	# else:
	#     st.write('Stopped')




	########################################################################################################################
	# import the necessary packages
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

	def detect_and_predict_mask(frame, faceNet, maskNet):
		# grab the dimensions of the frame and then construct a blob
		# from it
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
			(104.0, 177.0, 123.0))

		# pass the blob through the network and obtain the face detections
		faceNet.setInput(blob)
		detections = faceNet.forward()

		# initialize our list of faces, their corresponding locations,
		# and the list of predictions from our face mask network
		faces = []
		locs = []
		preds = []

		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with
			# the detection
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by ensuring the confidence is
			# greater than the minimum confidence
			if confidence > 0.5:
				# compute the (x, y)-coordinates of the bounding box for
				# the object
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# ensure the bounding boxes fall within the dimensions of
				# the frame
				(startX, startY) = (max(0, startX), max(0, startY))
				(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

				# extract the face ROI, convert it from BGR to RGB channel
				# ordering, resize it to 224x224, and preprocess it
				face = frame[startY:endY, startX:endX]
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)

				# add the face and bounding boxes to their respective
				# lists
				faces.append(face)
				locs.append((startX, startY, endX, endY))

		# only make a predictions if at least one face was detected
		if len(faces) > 0:
			# for faster inference we'll make batch predictions on *all*
			# faces at the same time rather than one-by-one predictions
			# in the above `for` loop
			faces = np.array(faces, dtype="float32")
			preds = maskNet.predict(faces, batch_size=32)

		# return a 2-tuple of the face locations and their corresponding
		# locations
		return (locs, preds)



	# load our serialized face detector model from disk
	print("[INFO] loading face detector model...")
	prototxtPath = r"C:\Users\Subhash\Desktop\Personal\Finalyearproject\FINAL-TRACKER-BOTH-MASK-SOCIAL\face_detector\deploy.prototxt"
	#print(prototxtPath)
	weightsPath = r"C:\Users\Subhash\Desktop\Personal\Finalyearproject\FINAL-TRACKER-BOTH-MASK-SOCIAL\face_detector\res10_300x300_ssd_iter_140000.caffemodel"
	faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

	# load the face mask detector model from disk
	print("[INFO] loading face mask detector model...")
	maskNet = load_model("mask_detector.model")

	# initialize the video stream and allow the camera sensor to warm up
	print("[INFO] starting video stream...")
	# vs = VideoStream(src=0).start()
	# time.sleep(2.0)





	##################################################################################################





	vs5 = cv2.VideoCapture("output-mask-social-1.mp4")
	vs6 = cv2.VideoCapture("output-mask-social-2.mp4")
	vs7 = cv2.VideoCapture("output-mask-social-3.mp4")




	######################################################################################################################################
	while run:
		###mask and social distancing
		ret, frame4 =  vs5.read()
		frame4 = cv2.cvtColor(frame4, cv2.COLOR_BGR2RGB)
		FRAME_WINDOW3.image(frame4,caption='Mask Detection and Social Distancing Monitoring')
		# if the `q` key was pressed, break from the loop
		

	#################################################################################################################################

	##############################################################################################################################################
		if config.Thread:
			frame = cap.read()

		else:
			(grabbed, frame) = vs.read()
			# if the frame was not grabbed, then we have reached the end of the stream
			if not grabbed:
				break
		
		frame = imutils.resize(frame, width=700)

		# detect faces in the frame and determine if they are wearing a
		# face mask or not
		(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

		# loop over the detected face locations and their corresponding
		# locations
		for (box, pred) in zip(locs, preds):
			# unpack the bounding box and predictions
			(startX, startY, endX, endY) = box
			(mask, withoutMask) = pred

			# determine the class label and color we'll use to draw
			# the bounding box and text
			label = "Mask" if mask > withoutMask else "No Mask"
			color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

			# include the probability in the label
			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

			# display the label and bounding box rectangle on the output
			# frame
			cv2.putText(frame, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

		# show the output frame
		#cv2.imshow("Frame", frame)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		FRAME_WINDOW.image(frame,caption='Real time Face Mask Detection')
		key = cv2.waitKey(1) & 0xFF
	#########################################################################################################################
	# loop over the frames from the video stream
		# read the next frame from the file
		if config.Thread:
			frame1 = cap.read()

		else:
			(grabbed, frame1) = vs.read()
			# if the frame was not grabbed, then we have reached the end of the stream
			if not grabbed:
				break

		# resize the frame and then detect people (and only people) in it
		frame = imutils.resize(frame1, width=700)
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
		FRAME_WINDOW2.image(frame,caption='Real time Social Distancing Monitoring')



	    # update the FPS counter
		#fps.update()

		# if an output video file path has been supplied and the video
		# writer has not been initialized, do so now
		

	# stop the timer and display FPS information
	# fps.stop()
	# print("===========================")
	# print("[INFO] Elasped time: {:.2f}".format(fps.elapsed()))
	# print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))

	# # close any open windows
	# cv2.destroyAllWindows()
		

	else:
		st.write('Stopped')