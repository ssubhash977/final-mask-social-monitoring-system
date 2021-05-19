def about_us():
	import streamlit as st 
	st.title('About Us')
	col1, col2 ,col3 = st.beta_columns(3)
	col1.image('subhash.png',caption="Subhash S")
	col2.image('sneha.png',caption="Sneha Korgal")
	col3.image('ullas.png',caption="Ullas A")
