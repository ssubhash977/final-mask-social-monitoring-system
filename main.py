import cv2
import streamlit as st

import final_streamlit_webapp


import both, mask, social, covid_track_app, about
st.set_page_config(layout="wide")
st.sidebar.title('Social Distancing and Mask Detection Monitoring System')
choice = st.sidebar.selectbox("Go to", list(["Choose a System","Home Dashboard","Both Monitoring system","Mask Detection","Social Distancing","Covid-19 cases Tracker","About Us"]))

if choice == 'Choose a System':
	st.title("Welcome to Social Distancing and Mask Detection Monitoring System")
elif choice == 'Home Dashboard':
    final_streamlit_webapp.all_functions()
elif choice == 'Both Monitoring system':
    both.both_run()
elif choice == 'Mask Detection':
    mask.mask_det()
elif choice == 'Social Distancing':
    social.social_dist()
elif choice == 'Covid-19 cases Tracker':
    covid_track_app.covid()
elif choice == 'About Us':
	about.about_us()
