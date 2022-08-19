import streamlit as st 
from lib import *
import numpy as np
# from streamlit_toggle import st_toggleswitch

st.set_page_config(layout='wide', page_title='Telecom KPI forecasting')

data_frames = []
technologies = ['LTE', '3G', '2G']

def diff(state=True):
	st.session_state['diff_button'] = state
	st.experimental_rerun()

def main() :
	
	st.title('Telecom KPI forecasting')

	# Reading data frame
	path = st.text_input("Enter Data Frame Path :")
	
	if path:
		try :
			df = get_data_frame(path)
			if "DataFrames" not in st.session_state :
				st.session_state['DataFrames'] = {}
			if path not in st.session_state['DataFrames'] :
				st.session_state['DataFrames'][path] = df
		except FileNotFoundError :
			st.error("Invalid path name !")
		except :
			print('other problems !')

	
	# Get data frames
	if 'DataFrames' in st.session_state :
		for data_frame in st.session_state['DataFrames'] :
			data_frames.append(data_frame)

	# Initialize data frame selection
	if "df_name" in st.session_state :
		data_frame = st.selectbox("Select Data Frame", data_frames, index=data_frames.index(st.session_state["df_name"]))
	else:
		data_frame = st.selectbox("Select Data Frame", data_frames)
	# Get session state
	if len(data_frames) :
		st.session_state['df_name'] = data_frame
	
	# Inialize technology selection
	if 'technology' in st.session_state :
		technology = st.selectbox('Select Technology', technologies, index=technologies.index(st.session_state['technology']))
	else :
		technology = st.selectbox('Select Technology', technologies)
	# Get technology
	st.session_state['technology'] = technology


	# Apply differencing
	if 'diff_button' not in st.session_state :
		st.session_state['diff_button'] = False

	if not st.session_state['diff_button'] :
		st.write("Apply differencing")
		col1, col2, col3, col4, col5  = st.columns(5, gap='small')
		button1 = col1.button('Enable differencing')
		button2 = col2.button('Disable differencing', disabled=True)
		st.session_state['diff_order'] = 0
	else :
		st.write("Apply differencing")
		col1, col2, col3, col4, col5  = st.columns(5, gap='small')
		button1 = col1.button('Enable differencing', disabled=True)
		button2 = col2.button('Disable differencing')
		st.session_state['diff_order'] = st.slider("Diffrenecing order", key='slider', value=st.session_state['diff_order'])

	
	if button1:
		diff()
	if button2:
		diff(False)

	# Save used data frame into session
	if "DataFrames" in st.session_state :
		df = get_technology(st.session_state['DataFrames'][data_frame],
													   technology)
		df.columns = [st.session_state["technology"]]
		st.session_state["DataFrame"] = df
	
if __name__=='__main__':
	main()
