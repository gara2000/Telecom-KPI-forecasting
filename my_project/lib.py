import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.eval_measures import rmse
from statsmodels.tsa.api import ExponentialSmoothing

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
import inspect

from tensorflow import keras
from keras.layers import Dense, LSTM, Conv1D, ConvLSTM1D
from keras.models import Sequential
from keras import Input

###############################################################
# Components Initializers
###############################################################

def rerun():
    st.experimental_rerun()

# REFRESH ON CHANGE 
def smooth(state_name, var):
    if state_name in st.session_state:
        old_state = st.session_state[state_name]
    else :
        old_state = var

    st.session_state[state_name]=var

    if old_state!=st.session_state[state_name]:
        rerun()

def number_input_initializer(title, min_value, max_value, init_value, state_name, step=1, Type=int):
    if state_name not in st.session_state:
        var = Type(st.number_input(title, min_value=min_value,
            max_value=max_value, value=init_value, step=step))
    else:
        var = Type(st.number_input(title, min_value=min_value, 
            max_value=max_value, value=Type(st.session_state[state_name]), step=step))
    
    smooth(state_name, var)

def selectbox_initializer(title, options, init_value, state_name):
    if state_name not in st.session_state:
        var = st.selectbox(title, options=options, index=init_value)
    else :
        var = st.selectbox(title, options=options, index=options.index(st.session_state[state_name]))

    smooth(state_name, var)

def slider_initializer(title, init_value, min_value, max_value, state_name):
    if state_name not in st.session_state:
        var = st.slider(title, value=init_value, min_value=min_value, max_value=max_value)
    else:
        var = st.slider(title, value=st.session_state[state_name], min_value=min_value, max_value=max_value)
    
    smooth(state_name, var)


#################################################################

# Page 1 : Data Frames
@st.cache
def get_data_frame(path):
    df = pd.read_csv(path)
    df.drop(df.columns[0], axis=1, inplace=True)
    date = df.columns[0]
    df[date]= pd.to_datetime(df[date])
    # time.sleep(4)
    return df

@st.cache
def get_technology(df, technology):
    filt = (df['RAT']==technology)
    new_df =  df.loc[filt]
    new_df = new_df.sort_values(by='Period')
    new_df.reset_index(drop=True, inplace=True)
    new_df.drop("RAT", axis=1, inplace=True)
    new_df.set_index("Period", inplace=True)
    return new_df

# @st.cache
def scale(df, technology):
    res = df.loc[:,[technology]]
    scaler = MinMaxScaler().fit(res)
    res = scaler.transform(res)
    df = pd.DataFrame(res, index=df.index)
    return df, scaler

#################################################################

# Page 2 : Plots & tests

def plot_data(ts):
    st.line_chart(ts)


def display_df(df):
    col1, col2 = st.columns(2)
    col1.subheader("Data :")
    col1.write(df)
    col2.subheader("Data Describtion :")
    col2.write(df.describe())

@st.cache
def create_df_from_ts(ts_list):
    df = pd.DataFrame({})
    df.index = ts_list[0].index
    for ts in ts_list :
        df[ts.name] = ts
    return df.dropna()


def test_stationarity(ts):
    
    # Determing rolling statistics
    rolmean = ts.rolling(window = 12, center = False).mean()
    rolstd = ts.rolling(window = 12, center = False).std()
    
    rolmean.name = 'Rolling average'
    rolstd.name = 'Rolling std'
    df = create_df_from_ts((ts, rolmean, rolstd))

    col1, col2 = st.columns([2,1])
    col1.line_chart(df, y=(ts.name, rolmean.name, rolstd.name))
    
    # Perform Dickey-Fuller test:
    # Null Hypothesis (H_0): time series is not stationary
    # Alternate Hypothesis (H_1): time series is stationary
    # print('Results of Dickey-Fuller Test:')
    dftest = adfuller(ts, 
                      autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], 
                         index = ['Test Statistic',
                                  'p-value',
                                  '# Lags Used',
                                  'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    col2.write(dfoutput)
    return dfoutput

# @st.cache
def plot_acf_pacf(ts, nb_lags, technology):
    pacf_vals = pacf(ts, nlags=nb_lags)
    acf_vals = acf(ts, nlags=nb_lags)

    f1, ax = plt.subplots()
    ax.bar(range(nb_lags), pacf_vals[:nb_lags])
    ax.set_title("PACF for {} Time-Series".format(technology))

    f2, ax = plt.subplots()
    ax.bar(range(nb_lags), acf_vals[:nb_lags])
    ax.set_title("ACF for {} Time-Series".format(technology))
    return f1, f2

def seasonal_component(ts, model='additive', extrapolate_trend='freq', plot=True):
    result = seasonal_decompose(ts, model=model, extrapolate_trend=extrapolate_trend)
    fig = result.plot()
    # st.pyplot(fig)
    return fig, result.seasonal

#################################################################

# Page 3 : Choosing the model

### Machine learning models

#feature engineering
@st.cache(allow_output_mutation=True)
def create_features(df, nlags, technology):
    res = df.copy()
    for i in range(1,nlags+1):
        res['lag'+str(i)] = res[technology].shift(i)
    return res.iloc[:, range(nlags+1)[::-1]]

def drop_features(df, nlags):
    for i in range(1,nlags+1):
        df.drop('lag'+str(i), axis=1, inplace=True)

def get_hyperparams(model):
	return inspect.signature(model.__init__)

def grid_search(model, X_train, y_train, params, cv, verbose, return_train_score):
    start = time.time()
    grid = GridSearchCV(estimator = model, param_grid = params, cv = cv, verbose = verbose, return_train_score = return_train_score, error_score="raise")
    grid.fit(X_train, y_train)
    return grid.best_params_, grid.best_score_, time.time()-start


### Deep learning models
# @st.cache
def NeuralNet(layers, optimizer, loss):
	model = Sequential()

	for layer in layers :
		model.add(layer)

	model.compile(optimizer=optimizer, loss=loss)
	return model

#################################################################

# Page 4 : Training the model

### Statistical models
# @st.cache
def fit_SARIMAX(ts_train, order, seasonal_order):
	model = SARIMAX(ts_train, order=order, seasonal_order=seasonal_order)

	start = time.time()

	model_fit = model.fit()

	return model_fit, time.time()-start

# @st.cache
def fit_Exp(ts_train, seasonal_period, seasonal, trend=None):
    start = time.time()
    model = ExponentialSmoothing(ts_train , trend=trend, seasonal_periods=seasonal_period, seasonal=seasonal).fit()

    return model, time.time()-start

### Machine learning models
def fit_XGBRegressor(X_train, y_train, learning_rate=0.1, n_estimators=300, max_depth=3, random_state=0):
    start = time.time()
    model = XGBRegressor(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model, time.time()-start

def fit_RandomForestRegressor(X_train, y_train, criterion='mse', n_estimators=210, max_depth=None, random_state=0):
    start = time.time()
    model = RandomForestRegressor(criterion=criterion, n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    return model, time.time()-start

### Deep learning models
# @st.cache
def fit_NN(model, X_train, y_train, validation_split=0.1, epochs=5):
    start = time.time()

    model.fit(X_train, y_train, validation_split=validation_split, epochs=epochs, shuffle=False)

    return model, time.time()-start

#################################################################

# Page 5 : Forecasting
# @st.cache
def statsmodels_forecasting(model, n_steps):
	return model.forecast(n_steps)

# this function forecasts the time series for n_steps hours in the future
# @st.cache
def ML_forecasting(model, n_steps, y_train, nlags):
    keys = ['lag'+str(i) for i in range(1,nlags+1)]
    values = list(y_train[y_train.shape[0]-1:y_train.shape[0]-1-nlags:-1])
    data = {keys[i]:[values[i]] for i in range(nlags)}
    X = pd.DataFrame(data)
    y_pred = []
    
    for t in range(n_steps):
        y = model.predict(X)
        y_pred.append(y[0])
        new = y[0]
        X.iloc[0,1:X.shape[1]] = X.iloc[0,0:X.shape[1]-1]
        X.iloc[0,0] = new
    
    return y_pred

# @st.cache
def LSTM_forecasting(model, n_steps, y_train, nlags):
    
    y_pred = []
    batch = np.array(y_train[-nlags:])
    batch = batch.reshape((1,nlags,1))

    for i in range(n_steps):
        y = model.predict(batch, verbose=0)[0]
        y_pred.append(y)
        batch = np.append(batch[:,1:,:], [[y]], axis=1)
        
    return y_pred

@st.cache
def nrmse(actual, prediction):
	return rmse(actual, prediction)/(np.max(actual)-np.min(actual))

def visual_comparison(actual, prediction, title):
    fig = plt.figure(figsize=(20,6))
    plt.plot(actual)
    plt.plot(prediction)
    plt.legend(('Actual', 'Prediction'), loc='best')
    plt.title(title)
    st.pyplot(fig)
    return fig




#################################################################

# Annexe page

### Statistical models
@st.cache
def split_data(df, delim):
	train = df.iloc[:delim]
	test = df.iloc[delim:]
	return train, test

### Machine learning models
# @st.cache
def ML_split(data, nlags, test_size, technology):
    X = data[['lag'+str(i) for i in range(1,nlags+1)]]
    y = data[technology]
    if test_size>0:
        return train_test_split(X, y, test_size=test_size, random_state=0, shuffle=False)
    return X,np.array([]),y,np.array([])

### Deep learning models
#Formatting data
@st.cache
def formatting(df, window_size):
    X = []
    y = []
    for i in range(df.shape[0]-window_size):
        X.append([[a] for a in list(df.iloc[i:i+window_size,0])])
        y.append(df.iloc[i+window_size,0])
    return np.array(X), np.array(y)