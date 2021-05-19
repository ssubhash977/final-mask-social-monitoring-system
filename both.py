import cv2

import streamlit as st

def both_run():
	st.header("Social Distancing and Mask Detection Monitoring System")
	run = st.checkbox('Run')

	vs5 = cv2.VideoCapture("output-mask-social-1.mp4")
	vs6 = cv2.VideoCapture("output-mask-social-2.mp4")
	vs7 = cv2.VideoCapture("output-mask-social-3.mp4")

	st.sidebar.header('Bounding Box Conditions')
	st.sidebar.image('conditions.jpeg')







	FRAME_WINDOW3= st.image([])
	FRAME_WINDOW4= st.image([])
	FRAME_WINDOW5= st.image([])
	FRAME_WINDOW6= st.image([])
	img = 'conditions.jpeg'
	while run:
		###mask and social distancing
		ret, frame4 =  vs5.read()
		ret, frame5 =  vs6.read()
		ret, frame6 =  vs7.read()
		frame4 = cv2.cvtColor(frame4, cv2.COLOR_BGR2RGB)
		frame5 = cv2.cvtColor(frame5, cv2.COLOR_BGR2RGB)
		frame6 = cv2.cvtColor(frame6, cv2.COLOR_BGR2RGB)
		FRAME_WINDOW3.image(frame4,caption='Mask Detection and Social Distancing Monitoring')
		FRAME_WINDOW4.image(frame5,caption='Mask Detection and Social Distancing Monitoring')
		FRAME_WINDOW5.image(frame6,caption='Mask Detection and Social Distancing Monitoring')
		
		FRAME_WINDOW6.image(img,caption='Bounding Box Conditions',width = 500)

	else:
		st.write('Stopped')
