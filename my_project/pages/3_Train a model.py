from lib import *
import streamlit as st

st.set_page_config(layout='wide')

# if "test" not in st.session_state:
# 	st.session_state["test"]=False

# if st.session_state["test"]:
# 	st.session_state["test"]=False
# 	st.experimental_rerun()



###############################################################

def idiate():
	if 'id' not in st.session_state:
		st.session_state['id']=0
	else :
		st.session_state['id']+=1

def change_state(state):
	st.session_state["state"]=state
	st.experimental_rerun()

def switch(col, key):
	if col.button("Change model", key=key) :
		change_state('Initial')


def initiate(f, text, init_val, key=0, Type=str):
	try :
		if key==0 :
			key = text[-1]
		if key in st.session_state :
			val = f(text, value=st.session_state[key])
		else :
			val = f(text, value=init_val)
		# if key in st.session_state:
		# 	old_state = st.session_state[key]
		st.session_state[key]=Type(val)
		# if old_state != st.session_state[key] :
		# 	f()
	except :
		st.error("Invalid type !")

#############################################################
#SARIMAX
#############################################################

def train_SARIMAX(col, key):
	if col.button("Train model", key=key):
		if 'DataFrame' in st.session_state :
			df = st.session_state["DataFrame"]
			df.columns = [st.session_state["technology"]]
			ts = df[st.session_state["technology"]]
			n_test = int(len(ts)*st.session_state['test_split']/100)
			ts_train = ts[0:len(ts)-n_test]
			ts_test = ts[len(ts)-n_test:]

			idiate()

			info = {
				'id':st.session_state["id"],
				'first-training date':ts_train.index[0],
				'last-training date':ts_train.index[-1],
				'number of training hours':len(ts_train),
				'test_split':st.session_state['test_split'],
				'order':st.session_state['order'],
				'seasonal order':st.session_state['seasonal order']}
			with st.spinner("Fitting the model...") :
				model, t = fit_SARIMAX(ts_train, st.session_state['order'], st.session_state['seasonal order'])

			info['Training time']=t

			st.success("Model trained successfully in {} seconds".format(round(t, 3)))

			# Score
			if n_test:
				with st.spinner("Evaluating model :"):
					ts_pred = statsmodels_forecasting(model, n_test)
					info['nrmse'] = nrmse(ts_test, ts_pred)
					st.write("score : ", info['nrmse'])
					info['status'] = "Test model"
					info['actual/prediction'] = (ts_test, ts_pred)
					visual_comparison(ts_test, ts_pred, "Results of the SARIMAX model")
			else :
				info['status'] = 'Final model'

			if "Trained models" not in st.session_state :
				st.session_state['Trained models'] = {"SARIMAX":[(model, info)]}
			else :
				if "SARIMAX" not in st.session_state['Trained models']:
					st.session_state["Trained models"]["SARIMAX"] = [(model, info)]
				else:
					st.session_state["Trained models"]["SARIMAX"].append((model, info))

			
		else :
			st.error("No Data Frame Selected")

def sarimax(key):
	st.title("SARIMAX model : hyperparameter tuning")

	# test split
	st.markdown("#### Test-data percentage :")
	# test_split = st.slider("set to 0 to train a final model on the whole data", value=20)
	initiate(st.slider, "Test split : (set to 0 to train a final model on the whole data)", 20, "test_split", int)

	# order
	st.markdown("#### Order : (p,d,q)")
	col2, col3, col4 = st.columns(3)
	initiate(col2.text_input, "Autor Regression parameter : p", 1, Type=int)
	initiate(col3.text_input, "Difference parameter : d", 0, Type=int)
	initiate(col4.text_input, "Moving average parameter : q", 1, Type=int)
	st.session_state['order'] = (st.session_state['p'],
		st.session_state['d'], st.session_state['q'])
	# seasonal order
	st.markdown("#### Seasonal order : (P,D,Q,M)")
	col2, col3, col4, col5 = st.columns(4)
	initiate(col2.text_input, "Autor Regression parameter : P", 1, Type=int)
	initiate(col3.text_input, "Difference parameter : D", 0, Type=int)
	initiate(col4.text_input, "Moving average parameter : Q", 1, Type=int)
	initiate(col5.text_input, "Periodicity parameter : M", 24, Type=int)
	st.session_state['seasonal order'] = (st.session_state['P'],
	 st.session_state['D'], st.session_state['Q'], st.session_state['M'])

	st.markdown("---")
	
	# validate or change
	col1, col2, col3, col4, col5 = st.columns(5, gap="small")
	train_SARIMAX(col1, key)
	switch(col2, key*10)

#############################################################
# ExponentialSmoothing
#############################################################

def train_exp(col, key):
	if col.button("Train model", key=key):
		if 'DataFrame' in st.session_state :
			df = st.session_state["DataFrame"]
			df.columns = [st.session_state["technology"]]
			ts = df[st.session_state["technology"]]
			n_test = int(len(ts)*st.session_state['test_split']/100)
			ts_train = ts[0:len(ts)-n_test]
			ts_test = ts[len(ts)-n_test:]

			idiate()
			
			info = {
				'id':st.session_state['id'],
				'first-training date':ts_train.index[0],
				'last-training date':ts_train.index[-1],
				'number of training hours':len(ts_train),
				'test_split':st.session_state['test_split'],
				'trend':st.session_state['trend'],
				'seasonal':st.session_state['seasonal'],
				'Seasonal period':st.session_state['seasonal period']}
			with st.spinner("Fitting the model...") :
				model, t = fit_Exp(ts_train, st.session_state['seasonal period'], st.session_state['seasonal'], st.session_state['trend'])

			info['Training time']=t

			st.success("Model trained successfully in {} seconds".format(round(t, 3)))

			# Score
			if n_test:
				with st.spinner("Evaluating model :"):
					ts_pred = statsmodels_forecasting(model, n_test)
					info['nrmse'] = nrmse(ts_test, ts_pred)
					st.write("score : ", info['nrmse'])
					info['status'] = "Test model"
					info['actual/prediction'] = (ts_test, ts_pred)
					visual_comparison(ts_test, ts_pred, "Results of the ExponentialSmooting model")
			else :
				info['status'] = 'Final model'

			if "Trained models" not in st.session_state :
				st.session_state['Trained models'] = {"ExponentialSmooting":[(model, info)]}
			else :
				if "ExponentialSmooting" not in st.session_state['Trained models']:
					st.session_state["Trained models"]["ExponentialSmooting"] = [(model, info)]
				else:
					st.session_state["Trained models"]["ExponentialSmooting"].append((model, info))

		else :
			st.error("No Data Frame Selected")

def exp(key):
	st.title("ExponentialSmooting model : hyperparameter tuning")
	
	initiate(st.slider, "Test split : (set to 0 to train a final model on the whole data)", 20, "test_split", int)

	types = ['add', 'mul', 'additive', 'multiplicative', None]
	

	if 'trend' not in st.session_state:
		trend = st.radio("Type of trend component", types, key=1, index=4)
	else :
		trend = st.radio("Type of trend component", types, key=1, index=types.index(st.session_state['trend']))
	st.session_state['trend'] = trend


	col1, col2 = st.columns(2)
	if 'seasonal' not in st.session_state:
		seasonal = col1.radio("Type of seasonal component", types, key=2, index=1)
	else :
		seasonal = col1.radio("Type of seasonal component", types, key=2, index=types.index(st.session_state['seasonal']))
	st.session_state['seasonal'] = seasonal

	if st.session_state['seasonal']==None:
		seasonal_periods = col2.text_input('Seasonal period', value='0')
	else :
		if 'Seasonal period' not in st.session_state :
			seasonal_periods = col2.text_input('Seasonal period', value=24)
		elif st.session_state['seasonal period']=='0':
			seasonal_periods = col2.text_input('Seasonal period', value=24)			
		else :
			seasonal_periods = col2.text_input('Seasonal period', value=st.session_state['seasonal period'])
	st.session_state['seasonal period'] = int(seasonal_periods)
	
	st.markdown("---")

	col1, col2, col3, col4, col5 = st.columns(5, gap="small")
	train_exp(col1, key)
	switch(col2, key*10)

#############################################################
# RandomForestRegressor
#############################################################

criterion_list = ("squared_error", "absolute_error", "poisson")

def int_value_setting_for_grid_search(text, maxi, mini, step, Type=int):
	st.subheader(text)
	col1, col2, col3 = st.columns(3)
	try :
		M = Type(col2.text_input("Max value :", value=maxi))
		m = Type(col1.text_input("Min value :", value=mini))
		s = Type(col3.text_input("Step value :", value=step))
		assert M>0
		assert m>0
		assert s>0
	except :
		if Type==int :
			st.error("Give positive integer values !")
		else :
			st.error("Give positive float values !")
	if m>M :
		st.error('Min value should be smaller than max value !')
	ans = list(np.array(range(int(m*100), int(M*100)+1, int(s*100)))/100)
	return [Type(i) for i in ans]

def criterion_setting_for_grid_search(text):
	st.subheader(text)
	if ("criterions" not in st.session_state) or (len(st.session_state['criterions'])==0):
		criterions = st.multiselect("Select criterions", criterion_list, criterion_list[0])
	else :
		criterions = st.multiselect("Select criterions", criterion_list, st.session_state['criterions'])
	st.session_state['criterions'] = criterions
	return criterions

def perform_grid_search(info, model, X_train, y_train):

	with st.expander('Perform a grid search') :
		params = {}
		params['n_estimators'] = int_value_setting_for_grid_search("Number of estimators :", 350, 50, 100)
		params['max_depth'] = int_value_setting_for_grid_search("Maximum depth :", 15, 1, 3)
		if 'criterion' in info :
			params['criterion'] = criterion_setting_for_grid_search("Criterion :")
			var = len(params['criterion'])
		else :
			params['learning_rate'] = int_value_setting_for_grid_search("Learning rate :", 1, 0.1, 0.1, Type=float)
			var = len(params['learning_rate'])
		cv = st.slider("Cross validation folds :", value=2, max_value=10, min_value=2)
		nb_comb =  len(params['n_estimators'])*len(params['max_depth'])*var*cv
		
		# for key, val in params.items() :
		# 	st.write(key, val)
		if st.button("Perform grid search") :
			if nb_comb==0 :
				st.error("Select at least one criterion !")
			elif nb_comb > 50 :
				st.warning("{} combinations ! This could take a while !!!".format(nb_comb))
				if st.button("Perform grid search anyway") :
					with st.spinner('Performing grid search :') :
						best_params, best_score, t = grid_search(model, X_train, y_train, params, cv, 3, False)
					st.success("Grid search successfully performed in {} seconds".format(t))
					st.write("best parameters", best_params)
					st.write("best score", best_score)
			else :
				st.write("{} combinations".format(nb_comb))
				with st.spinner('Performing grid search :') :
					best_params, best_score, t = grid_search(model, X_train, y_train, params, cv, 3, False)

				st.success("Grid search successfully performed in {} seconds".format(round(t,3)))
				st.write("best parameters", best_params)
				st.write("best score", best_score)


def train_rfr(col, key):
	if col.button("Train model", key=key):
		if 'DataFrame' in st.session_state :
			
			df = st.session_state["ML_df"]
			X_train, X_test, y_train, y_test = ML_split(df, st.session_state['nlags'], st.session_state['test_size'], st.session_state['technology'])

			idiate()

			info = {
				'id':st.session_state['id'],
				'number of lags':st.session_state['nlags'],
				'first-training date':X_train.index[0],
				'last-training date':X_train.index[-1],
				'number of training hours':X_train.shape[0],
				'test_split':st.session_state['test_split'],
				'criterion':st.session_state['criterion'],
				'n_estimators':st.session_state['n_estimators'],
				'max_depth':st.session_state['max depth'],
				'y_train':y_train
				}
			if st.session_state["no max depth"]:
				info['max_depth']=None

			with st.spinner("Fitting the model...") :
				model, t = fit_RandomForestRegressor(X_train, y_train, info['criterion'], info['n_estimators'], info['max_depth'])

			info['Training time']=t

			st.success("Model trained successfully in {} seconds".format(round(t, 3)))

			# Score
			n_test=X_test.shape[0]
			if n_test:
				with st.spinner("Evaluating model..."):
					y_pred = ML_forecasting(model, n_test, y_train, st.session_state['nlags'])
					y_pred = pd.DataFrame(y_pred, index=X_test.index).iloc[:,0]
					info['nrmse'] = nrmse(y_test, y_pred)
					st.write("score : ", info['nrmse'])
					info['status'] = "Test model"
					info['actual/prediction'] = (y_test, y_pred)
					visual_comparison(y_test, y_pred, "Results of the RandomForestRegressor model")
			else :
				info['status'] = 'Final model'

			if "Trained models" not in st.session_state :
				st.session_state['Trained models'] = {"RandomForestRegressor":[(model, info)]}
			else :
				if "RandomForestRegressor" not in st.session_state['Trained models']:
					st.session_state["Trained models"]["RandomForestRegressor"] = [(model, info)]
				else:
					st.session_state["Trained models"]["RandomForestRegressor"].append((model, info))

			# st.success("Model trained successfully in {} seconds".format(round(t, 3)))
		else :
			st.error("No Data Frame Selected")

def rfr(key):
	st.title("Random Forest Regressor :")
	st.subheader("Feature Engineering :")

	if 'DataFrame' in st.session_state:
		if 'nlags' not in st.session_state :
			nlags = st.slider("Set the number of lags : (number of past hours to use as features)", value=48, max_value=24*30)
		else :
			nlags = st.slider("Set the number of lags : (number of past hours to use as features)", value=st.session_state['nlags'], max_value=24*30)
		st.session_state['nlags']=nlags

		# initiate(st.slider, , 48, "nlags", int)
		
		df = create_features(st.session_state['DataFrame'], st.session_state['nlags'], st.session_state['technology'])
		df.dropna(inplace=True)
		st.session_state['ML_df'] = df

		st.subheader("Parameter tuning :")
		initiate(st.slider, "Test split : (set to 0 to train a final model on the whole data)", 20, "test_split", int)
		
		st.session_state['test_size'] = round(st.session_state['test_split']/100, 3)


		X_train, X_test, y_train, y_test = ML_split(df, st.session_state['nlags'], st.session_state['test_size'], st.session_state['technology'])
		
		info = ["criterion", "n_estimators", "max_depth"]


		perform_grid_search(info, RandomForestRegressor(), X_train, y_train)
		st.markdown("---")

	else :
		st.error("No Data Frame Selected")

	st.subheader("Model Training :")
	# criterion selectbox
	if 'criterion' not in st.session_state:
		criterion = st.selectbox("Criterion :", criterion_list)
	else :
		criterion = st.selectbox("Criterion :", criterion_list, criterion_list.index(st.session_state['criterion']))
	st.session_state['criterion'] = criterion

	# n_estimators slider
	if 'n_estimators' not in st.session_state:
		n_estimators = st.slider("Number of estimators :", min_value=10, max_value=500, value=100)
	else :
		n_estimators = st.slider("Number of estimators :", min_value=10, max_value=500, value=st.session_state['n_estimators'])
	st.session_state['n_estimators'] = n_estimators

	# max_depth 
	st.write("Maximum depth :")
	col1, col2 = st.columns([1, 5])
	if "no max depth" not in st.session_state:
		none = col1.checkbox("None", value=True)
	else :
		none = col1.checkbox("None", st.session_state["no max depth"])
	st.session_state["no max depth"] = none
	try :
		if st.session_state["no max depth"] :
			max_depth = int(col2.text_input("Max depth :", value="5", disabled=True))
		else :
			max_depth = int(col2.text_input("Max depth :",value=st.session_state["max depth"], disabled=False))
		assert max_depth>0
	except :
		st.error("Max depth should be a positive integer !")
	st.session_state["max depth"] = max_depth

	st.markdown("---")
	col1, col2, col3, col4, col5 = st.columns(5, gap="small")
	train_rfr(col1, key)
	switch(col2, key*10)

#############################################################
# XGBRegressor
#############################################################
def train_xgbr(col, key):
	if col.button("Train model", key=key):
		if 'DataFrame' in st.session_state :
			
# TRAIN TEST SPLIT
			df = st.session_state["ML_df"]
			X_train, X_test, y_train, y_test = ML_split(df, st.session_state['nlags'], st.session_state['test_size'], st.session_state['technology'])

			idiate()

# MACHINE LEARNING MODEL INFORMATION
			info = {
				'id':st.session_state['id'],
				'number of lags':st.session_state['nlags'],
				'first-training date':X_train.index[0],
				'last-training date':X_train.index[-1],
				'number of training hours':X_train.shape[0],
				'test_split':st.session_state['test_split'],
				'learning_rate':st.session_state['learning_rate'],
				'n_estimators':st.session_state['n_estimators'],
				'max_depth':st.session_state['max depth'],
				'y_train':y_train
				}
			if st.session_state["no max depth"]:
				info['max_depth']=None

# FIT THE MACHINE LEARNING MODEL
			with st.spinner("Fitting the model...") :
				model, t = fit_XGBRegressor(X_train, y_train, info['learning_rate'], info['n_estimators'], info['max_depth'])

			info['Training time']=t

			st.success("Model trained successfully in {} seconds".format(round(t, 3)))

# CALCULATE THE SCORE AND VISUALLY COMPARE RESULTS
			n_test=X_test.shape[0]
			if n_test:
				with st.spinner("Evaluating model..."):
					y_pred = ML_forecasting(model, n_test, y_train, st.session_state['nlags'])
					y_pred = pd.DataFrame(y_pred, index=X_test.index).iloc[:,0]
					info['nrmse'] = nrmse(y_test, y_pred)
					st.write("score : ", info['nrmse'])
					info['status'] = "Test model"
					info['actual/prediction'] = (y_test, y_pred)
					visual_comparison(y_test, y_pred, "Results of the XGBoostRegressor model")
			else :
				info['status'] = 'Final model'

# ADD MODEL TO THE LIST OF TRAINED MODELS
			if "Trained models" not in st.session_state :
				st.session_state['Trained models'] = {"XGBRegressor":[(model, info)]}
			else :
				if "XGBRegressor" not in st.session_state['Trained models']:
					st.session_state["Trained models"]["XGBRegressor"] = [(model, info)]
				else:
					st.session_state["Trained models"]["XGBRegressor"].append((model, info))

			
		else :
			st.error("No Data Frame Selected")

def xgbr(key):
	st.title("XG Boost Regressor :")
	# st.error("Under construction 🚧")
	st.subheader("Feature Engineering :")

	if 'DataFrame' in st.session_state:

# NUMBER OF LAGS
		if 'nlags' not in st.session_state :
			nlags = st.slider("Set the number of lags : (number of past hours to use as features)", value=48, max_value=24*30)
		else :
			nlags = st.slider("Set the number of lags : (number of past hours to use as features)", value=st.session_state['nlags'], max_value=24*30)
		st.session_state['nlags']=nlags


# PREPARE THE MACHINE LEARNING DATA FRAME		
		df = create_features(st.session_state['DataFrame'], st.session_state['nlags'], st.session_state['technology'])
		df.dropna(inplace=True)
		st.session_state['ML_df'] = df

		st.subheader("Parameter tuning :")

# TEST SPLIT AND TEST SIZE
		initiate(st.slider, "Test split : (set to 0 to train a final model on the whole data)", 20, "test_split", int)
		
		st.session_state['test_size'] = round(st.session_state['test_split']/100, 3)

# TRAIN TEST SPLIT AND GRID SEARCH
		X_train, X_test, y_train, y_test = ML_split(df, st.session_state['nlags'], st.session_state['test_size'], st.session_state['technology'])
		
		info = ["learning_rate", "n_estimators", "max_depth"]


		perform_grid_search(info, XGBRegressor(), X_train, y_train)
		st.markdown("---")

	else :
		st.error("No Data Frame Selected")

	st.subheader("Model Training :")

# LEARNING RATE CHECKBOX
	if 'learning_rate' not in st.session_state:
		learning_rate = st.number_input("learning_rate :", min_value=0., max_value=1., step=0.01, value=0.1)
	else :
		learning_rate = st.number_input("learning_rate :", min_value=0., max_value=1., step=0.01, value=st.session_state['learning_rate'])
	st.session_state['learning_rate'] = learning_rate

# NUMBER OF ESTIMATORS SLIDER
	if 'n_estimators' not in st.session_state:
		n_estimators = st.slider("Number of estimators :", min_value=10, max_value=500, value=100)
	else :
		n_estimators = st.slider("Number of estimators :", min_value=10, max_value=500, value=st.session_state['n_estimators'])
	st.session_state['n_estimators'] = n_estimators

# MAXIMUM DEPTH
	# st.write("Maximum depth :")
	col1, col2 = st.columns([1, 5])
	if "no max depth" not in st.session_state:
		none = col1.checkbox("None", value=True)
	else :
		none = col1.checkbox("None", st.session_state["no max depth"])
	st.session_state["no max depth"] = none
	try :
		if st.session_state["no max depth"] :
			max_depth = int(col2.text_input("Max depth :", value="5", disabled=True))
		else :
			max_depth = int(col2.text_input("Max depth :",value=st.session_state["max depth"], disabled=False))
		assert max_depth>0
	except :
		st.error("Max depth should be a positive integer !")
	st.session_state["max depth"] = max_depth

	st.markdown("---")

# TRAINING XGBR
	col1, col2, col3, col4, col5 = st.columns(5, gap="small")
	train_xgbr(col1, key)
	switch(col2, key*10)

#############################################################
# Neural Network
#############################################################

activation_functions = ["linear", "relu", "sigmoid", "softmax", "tanh"]
layers_list = ["FC", "LSTM", "CNN"]


# ADD LAYERS TO THE NN MODEL
def add_layer(layer_type, name):

# CNN LAYER ADDING
	if layer_type=='CNN':
		
		number_input_initializer("Number of filters :",
			1, 500, 30, "filters", 1)
		number_input_initializer("Filter size :",
			1, st.session_state['nlags'], 3, "filter_size", 1)
		number_input_initializer("Strides :",
			1, int(st.session_state['nlags']/2), 1, "strides", 1)

		info = {
			"Layer":"Convolutional",
			"Number of filters":st.session_state['filters'],
			"Filter size":st.session_state["filter_size"],
			"Strides":st.session_state["strides"],
			"Padding":"same"
		}
		layer = Conv1D(filters=info["Number of filters"],
			kernel_size=info["Filter size"],
			strides=info["Strides"],
			padding=info["Padding"],
			name=name
			)
		if st.button("Add Convolutional layer") :
			st.session_state["layers_info"].append(info)
			st.session_state["layers"].append(layer)


# LSTM LAYER ADDING
	elif layer_type=='LSTM':
		
		number_input_initializer("Number of units :", 10, 500, 100, "units", 1)
		selectbox_initializer("Activation function :", activation_functions, 0, "activation")
		selectbox_initializer("Recurrent activation function :", activation_functions, 0, "re_activation")
		number_input_initializer("Dropout probability :",
			0., 1., 0., "dropout", 0.01, float)
		number_input_initializer("Recurrent dropout probability :",
			0., 1., 0., "re_dropout", 0.01, float)

		info = {
			"Layer":"LSTM",
			"Number of units":st.session_state["units"],
			"Activation function":st.session_state["activation"],
			"Recurrent activation function":st.session_state["re_activation"],
			"Dropout":st.session_state["dropout"],
			"Recurrent dropout":st.session_state["re_dropout"]
		}
		layer = LSTM(units=st.session_state["units"], activation=st.session_state["activation"],
			recurrent_activation=st.session_state["re_activation"],
			dropout=st.session_state["dropout"],
			recurrent_dropout=st.session_state["re_dropout"],
			name=name
			)
		if st.button("Add LSTM layer") :
			st.session_state["layers_info"].append(info)
			st.session_state["layers"].append(layer)

# FC LAYER ADDING
	elif layer_type=='FC':
		
		number_input_initializer("Number of units :", 10, 500, 100, "units", 1)
		selectbox_initializer("Activation function :", activation_functions, 0, "activation")
		info = {
			"Layer":"FC",
			"Number of units":st.session_state["units"],
			"Activation function":st.session_state["activation"]
		}
		layer = Dense(units=info["Number of units"], activation=info["Activation function"],
			name=name)
		if st.button("Add fully connected layer") :
			st.session_state["layers_info"].append(info)
			st.session_state["layers"].append(layer)

	# elif layer_type=='CLSTM':
	# 	number_input_initializer("Number of filters :",
	# 		1, 500, 30, "filters", 1)
	# 	number_input_initializer("Filter size :",
	# 		1, st.session_state['nlags'], 3, "filter_size", 1)
	# 	number_input_initializer("Strides :",
	# 		1, int(st.session_state['nlags']/2), 1, "strides", 1)
	# 	# number_input_initializer("Number of units :", 10, 500, 100, "units", 1)
	# 	selectbox_initializer("Activation function :", activation_functions, 0, "activation")
	# 	selectbox_initializer("Recurrent activation function :", activation_functions, 0, "re_activation")
	# 	number_input_initializer("Dropout probability :",
	# 		0., 1., 0., "dropout", 0.01, float)
	# 	number_input_initializer("Recurrent dropout probability :",
	# 		0., 1., 0., "re_dropout", 0.01, float)

	# 	info = {
	# 		"Layer":"CLSTM",
	# 		"Activation function":st.session_state["activation"],
	# 		"Recurrent activation function":st.session_state["re_activation"],
	# 		"Dropout":st.session_state["dropout"],
	# 		"Recurrent dropout":st.session_state["re_dropout"],
	# 		"Number of filters":st.session_state['filters'],
	# 		"Filter size":st.session_state["filter_size"],
	# 		"Strides":st.session_state["strides"],
	# 		"Padding":"same"
	# 	}
	# 	layer = ConvLSTM1D(
	# 		filters=info["Number of filters"],
	# 		kernel_size=info["Filter size"],
	# 		strides=info["Strides"],
	# 		padding=info["Padding"],
	# 		activation=st.session_state["activation"],
	# 		recurrent_activation=st.session_state["re_activation"],
	# 		dropout=st.session_state["dropout"],
	# 		recurrent_dropout=st.session_state["re_dropout"],
	# 		name=name
	# 		)
	# 	if st.button("Add convolutional LSTM layer") :
	# 		st.session_state["layers_info"].append(info)
	# 		st.session_state["layers"].append(layer)

# LIST OF USED LAYERS IN THE NN MODEL
def used_layers():
	s = []
	for info in st.session_state["layers_info"]:
		s.append(info["Layer"])
	return s



### NEURAL NET MODEL TRAINING ###
def train_nn(col, key):
	if col.button("Train model", key=key):
		if 'DataFrame' in st.session_state :
			if "LSTM" not in used_layers() :
				st.error("Use an LSTM layer !")
				return
			
# SCALING AND TRAIN TEST SPLIT
			df, scaler = scale(st.session_state["DataFrame"], st.session_state["technology"])

			st.session_state["scaler"] = scaler

			X, y= formatting(df, st.session_state['nlags'])

			n_test = int((st.session_state["test_split"]/100)*X.shape[0])

			X_train = X[:X.shape[0]-n_test]
			y_train = y[:X.shape[0]-n_test]
			y_test = y[-n_test:]
			# y_test
			y_indexed = st.session_state["DataFrame"].iloc[:X.shape[0]-n_test]
			indexed = st.session_state["DataFrame"].iloc[-n_test:]

			idiate()

# NN MODEL INFORMATION
			info = {
				'id':st.session_state['id'],
				'number of lags':st.session_state['nlags'],
				'number of training hours':X_train.shape[0],
				'test_split':st.session_state['test_split'],
				'layers information':st.session_state["layers_info"],
				'y_train':pd.DataFrame(y_train, index=y_indexed.index)
				}
			layers = st.session_state["layers"].copy()
			layers.append(Dense(1, name="unique"))

# TRAINING THE NN MODEL
			with st.spinner("Fitting the model...") :
				model = NeuralNet(layers, "adam", "mse")
				model, t= fit_NN(model, X_train, y_train, epochs=st.session_state["epochs"])
			
			info['Training time']=t

			st.success("Model trained successfully in {} seconds".format(round(t, 3)))

# CALCULATE SCORE AND VISUALLY COMPARE RESULTS
			if n_test:
				with st.spinner("Evaluating model..."):
					y_pred = LSTM_forecasting(model, n_test, y_train, st.session_state['nlags'])
					st.write(y_pred)
					y_pred = pd.DataFrame(y_pred, index=indexed.index).iloc[:,0]
					y_test = pd.DataFrame(y_test, index=indexed.index).iloc[:,0]
					info['nrmse'] = nrmse(y_test, y_pred)
					st.write("score : ", info['nrmse'])
					info['status'] = "Test model"
					info['actual/prediction'] = (y_test, y_pred)
					visual_comparison(y_test, y_pred, "Results of the Neural Network model")
			else :
				info['status'] = 'Final model'

# ADD THE NN MODEL TO THE TRAINED MODELS LIST
			if "Trained models" not in st.session_state :
				st.session_state['Trained models'] = {"LSTM Neural Network":[(model, info)]}
			else :
				if "LSTM Neural Network" not in st.session_state['Trained models']:
					st.session_state["Trained models"]["LSTM Neural Network"] = [(model, info)]
				else:
					st.session_state["Trained models"]["LSTM Neural Network"].append((model, info))

			# st.success("Model trained successfully in {} seconds".format(round(t, 3)))
		else :
			st.error("No Data Frame Selected")


#### NEURAL NETWORK INTERFACE ####
def nn(key):
	st.title("LSTM Neural Network :")
	# st.error("Under construction 🚧")

# NUMBER OF LAGS
	slider_initializer("Number of lags :", 48, 1, 100, 'nlags')

# TEST SPLIT
	slider_initializer("Test split : (set to 0 to train a final model on the whole data)",
		20, 0, 100, 'test_split')

# INPUT SHAPE AND INPUT LAYER ADDING
	st.session_state["input_shape"]=(st.session_state['nlags'], 1)

	info = {
		"Layer":"Input",
		"Input shape":st.session_state["input_shape"]
	}

	if "layers_info" not in st.session_state:
		st.session_state["layers_info"]=[info]
	else:
		st.session_state["layers_info"][0]=info

	if "layers" not in st.session_state:
		st.session_state["layers"]=[Input(shape=st.session_state["input_shape"])]
	else :
		st.session_state["layers"][0]=Input(shape=st.session_state["input_shape"])
		

# ADD LAYER
	with st.expander("Add layer :"):
		selectbox_initializer("Choose layer :", layers_list, 1, "chosen_layer")
		list_of_used_layers = used_layers()
		if len(list_of_used_layers)==1 and st.session_state["chosen_layer"]=='FC':
			st.error("Start with an LSTM or a CNN layer !")
		elif ("LSTM" in list_of_used_layers or "FC" in list_of_used_layers) and st.session_state["chosen_layer"]=='CNN':
			st.error("Use CNN layers in the beginning !")
		elif (st.session_state["chosen_layer"] in ["LSTM", "CLSTM"]) and ("FC" in list_of_used_layers):
			st.error("Use LSTM layers before FC layers !")
		else :
			add_layer(st.session_state["chosen_layer"],"layer"+str(len(st.session_state["layers"])))

# LAYERS DISPLAY
	st.subheader("Model :")
	nb = [1 for i in range(len(st.session_state["layers_info"]))]
	col = st.columns(nb)
	for i,info in enumerate(st.session_state["layers_info"]) :
		if i==1 :
			st.session_state["first_layer"]=info["Layer"]
		with col[i].expander("{} layer:".format(info["Layer"])):
			for key, val in info.items():
				if key!="Layer":
					st.markdown("###### {}:".format(key))
					st.write(val)

# REMOVE LAYERS
	if len(st.session_state["layers"])>1:
		but = st.button("Remove layer")
	else :
		but = st.button("Remove layer", disabled=True)

	if 'rem_msg' not in st.session_state:
		st.session_state['rem_msg']=False

	if st.session_state["rem_msg"] :
		st.session_state["rem_msg"]=False
		st.warning("Layer removed !")

	if but:
		removed = st.session_state["layers_info"].pop()
		st.session_state["layers"].pop()
		st.session_state["rem_msg"]=True
		rerun()

# NUMBER OF EPOCHS
	number_input_initializer(title="Number of epochs :", min_value=1, max_value=100, init_value=10, state_name="epochs", step=1, Type=int)

	st.markdown("---")
	col1, col2, col3, col4, col5 = st.columns(5, gap="small")
	train_nn(col1, key)
	switch(col2, key*10)

#############################################################

def choose_model(models, key):
	model_name = st.selectbox('Models', models.keys())
	if st.button("choose model", key=key):
		change_state(model_name)

def init(key):
	tab1, tab2, tab3 = st.tabs(["Statistical models",
		"Machine learning models", "Deep learning models"])

	with tab1:
		models = {
			'SARIMAX':sarimax,
			'ExponentialSmooting':exp
			}

		choose_model(models, 0)

	with tab2:
		models = {
			'RandomForestRegressor' : rfr,
			'XGBoostRegressor' : xgbr
			}
		choose_model(models, 1)

	with tab3:
		models = {
			'LSTM' : nn
		}
		choose_model(models, 2)


###############################################################




# st.title('Training a forecasting model')

if "state" not in st.session_state :
	st.session_state["state"]="Initial"

states = {
	'Initial' : init,
	'SARIMAX': sarimax,
	'ExponentialSmooting' : exp,
	'RandomForestRegressor' : rfr,
	'XGBoostRegressor' : xgbr,
	'LSTM' : nn
}
keys = {
	'Initial' : 1,
	'SARIMAX': 2,
	'ExponentialSmooting' : 3,
	'RandomForestRegressor' : 4,
	'XGBoostRegressor' : 5,
	'LSTM' : 6
}

states[st.session_state["state"]](keys[st.session_state["state"]])


