import streamlit as st 
import pandas as pd
from lib import *

# st.title('Data visualization')

if 'DataFrame' in st.session_state:
	df = st.session_state['DataFrame']
	if 'technology' in st.session_state :
		df.columns = [st.session_state['technology']]
		df1 = df.copy()
		if st.session_state['diff_order']:
			df1[st.session_state['technology']] = df[st.session_state['technology']].diff(st.session_state['diff_order'])
		
		tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data description",
			"Data visualization", "Stationarity testing",
			"Trend & Seasonality", "PACF & ACF plots"
			])

		df1.dropna(inplace=True)	
		ts = df1[st.session_state['technology']]
		with tab1:	
			display_df(df1)

		with tab2:
			plot_data(ts)
		
		with tab3:
			st.subheader("Augmented Dikey-Fuller test")
			test_stationarity(ts)

		with tab4:
			fig, sc = seasonal_component(ts)
			st.pyplot(fig)


		with tab5:
			nb_lags = st.slider("Number of lags :")
			pacf, acf = plot_acf_pacf(ts, nb_lags, st.session_state['technology'])
			col1, col2 = st.columns(2)
			col1.pyplot(pacf)
			col2.pyplot(acf)
else :
	tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data description",
			"Data visualization", "Stationarity testing",
			"Trend & Seasonality", "PACF & ACF plots"
			])
	st.error('No Data Frame Selected')

