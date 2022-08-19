from lib import *
import streamlit as st

st.title("Predictions :")
if "predictions" in st.session_state :
	for prediction in st.session_state["predictions"]:
		with st.container():
			st.markdown("### Predictions of the {} model (model id : {})".format(prediction[0], prediction[1]))
			col1, col2 = st.columns([1,3])
			col1.write(prediction[2])
			col2.line_chart(prediction[2])
else :
	st.error("No predictions made !")