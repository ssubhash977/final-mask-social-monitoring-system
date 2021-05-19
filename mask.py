def detect_and_predict_mask(frame, faceNet, maskNet):
		# grab the dimensions of the frame and then construct a blob
		# from it
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
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))

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


def mask_det():
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
	
	#------------------------------------------------------------------------------#


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




	st.header("Real time Mask Detection Monitoring System")
	run1 = st.checkbox('Open Camera')
	FRAME_WINDOW= st.image([])
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
	
	while run1:
		if config.Thread:
				frame = cap.read()

		else:
			(grabbed, frame) = vs.read()
			# if the frame was not grabbed, then we have reached the end of the stream
		
			
		frame = imutils.resize(frame, width=900)

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
