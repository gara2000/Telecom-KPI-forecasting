import streamlit as st 
from lib import *

def trained_models():
	st.title("Trained models :")
	# ENSURE THERE IS SOME TRAINED MODELS
	if "Trained models" in st.session_state:
		# st.write(st.session_state['Trained models'])
		models_score = {}
		for model, l in st.session_state['Trained models'].items():
			# st.write(l)
			with st.expander(model) :

				for i,m in enumerate(l) :
					if m[1]['status']=="Test model":
						models_score["Model id = {}".format(m[1]['id'])] = [m[1]['nrmse']]
					ids[m[1]['id']]=(m[1]["status"], m[0], model)
					st.subheader("Model"+str(i+1))
					# st.write(m[0])
					st.subheader("Model description :")
					for key, val in m[1].items():
						if key=='actual/prediction' : 
							a = val[0]
							p = val[1]
							visual_comparison(a, p, model+" results :")
						else :
							st.write(key+": ", val)
		models_score = dict(sorted(models_score.items(), key=lambda item :item[1]))
		models_score['model id']=["Score"]
		out = pd.DataFrame(models_score)
		out.set_index(out["model id"], inplace=True)
		out.drop("model id", axis=1, inplace=True)
		st.table(out)

	else:
		st.error("No trained models !")


def forecast(idd, nsteps):
	model = ids[idd][1]
	model_name = ids[idd][2]
	with st.spinner("Forecasting..."):
# Forecasint LSTM NN models
		if model_name == "LSTM Neural Network":
			# st.write(st.session_state["Trained models"][model_name])
			for m in st.session_state["Trained models"][model_name]:
				# for m in models.values() :
				# st.write(m.layers)
				if m[1]["id"]==idd :
					nlags = m[1]["number of lags"]
					y_train = m[1]["y_train"]
					indexed = pd.date_range(y_train.index[-1], periods=st.session_state["nsteps"]+1, freq='H')
					# st.write(indexed)
					y_pred = LSTM_forecasting(model, st.session_state["nsteps"], y_train, nlags)
					y_pred = st.session_state["scaler"].inverse_transform(y_pred)
					y_pred = pd.DataFrame(y_pred, index=indexed[1:])
					y_pred.columns = [st.session_state["technology"]]
					
					if "predictions" not in st.session_state:
						st.session_state["predictions"]=[(model_name, idd, y_pred)]
					else:
						st.session_state["predictions"].append((model_name, idd, y_pred))
					st.subheader(model_name+" Predictions (model id {})".format(idd))
					st.write(y_pred)
					st.line_chart(y_pred)

# Forecasting ML models
		if model_name in ["XGBRegressor", "RandomForestRegressor"]:
			for m in st.session_state["Trained models"][model_name]:

				if m[1]["id"]==idd :
					nlags = m[1]["number of lags"]
					y_train = m[1]["y_train"]
					indexed = pd.date_range(y_train.index[-1], periods=st.session_state["nsteps"]+1, freq='H')
					y_pred = ML_forecasting(model, st.session_state["nsteps"], y_train, nlags)
					# y_pred = pd.DataFrame(y_pred, index=indexed.index).iloc[:,0]
					# y_pred = st.session_state["scaler"].inverse_transform(y_pred)
					y_pred = pd.DataFrame(y_pred, index=indexed[1:])
					y_pred.columns = [st.session_state["technology"]]
					
					if "predictions" not in st.session_state:
						st.session_state["predictions"]=[(model_name, idd, y_pred)]
					else:
						st.session_state["predictions"].append((model_name, idd, y_pred))
					st.subheader(model_name+" Predictions (model id {})".format(idd))
					st.write(y_pred)
					st.line_chart(y_pred)

# Forecasting Statistical models
		if model_name in ["ExponentialSmooting","SARIMAX"]:
			for m in st.session_state["Trained models"][model_name]:

				if m[1]["id"]==idd :
					y_pred = statsmodels_forecasting(model, st.session_state['nsteps'])
					y_pred = pd.DataFrame(y_pred)
					y_pred.columns = [st.session_state["technology"]]
					if "predictions" not in st.session_state:
						st.session_state["predictions"]=[(model_name, idd, y_pred)]
					else:
						st.session_state["predictions"].append((model_name, idd, y_pred))
					
					st.subheader(model_name+" Predictions (model id {})".format(idd))
					st.write(y_pred)
					st.line_chart(y_pred)
	

def forecasting():
	with st.expander("Forecasting :") :
		if "Trained models" in st.session_state:
			selectbox_initializer("Choose model by id", list(ids.keys()), 0, 'chosen_id')
			number_input_initializer("Number of steps to forecast :", 1, 24*365, 24*60, "nsteps")
			
			if st.button("Forecast"):
				forecast(st.session_state["chosen_id"], st.session_state["nsteps"])
		
		else :
			st.selectbox("Choose model by id:", [])

ids = {}

trained_models()
st.markdown('---')
forecasting()