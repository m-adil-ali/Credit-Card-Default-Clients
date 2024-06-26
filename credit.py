import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib



st.title("Credit Card Default Clients Prediction")
st.write("If you want to know more about the dataset (Default of Credit Card Clients), here is the [link](https://www.kaggle.com/code/gpreda/default-of-credit-card-clients-predictive-models)")

# Load the model from the file
model = joblib.load('data/Random_Forest_Classifiers.pkl')

# Load the StandardScaler from the file
scaler = joblib.load('data/scaler.pkl')

# Taking input from user and then using that input for prediction.
st.write("### Prediction Inputs")

with st.form(key= 'my_form'):
    LIMIT_BAL = st.number_input('Enter limit of your Credit Card', min_value=10000, max_value=1000000, value=100000)
    # SEX = st.number_input('SEX', min_value=1, max_value=2, value=1)
    # EDUCATION = st.number_input('EDUCATION', min_value=1, max_value=4, value=1)
    # MARRIAGE = st.number_input('MARRIAGE', min_value=1, max_value=3, value=1)
    # AGE = st.number_input('AGE', min_value=21, max_value=79, value=21)
    HIST_9 = st.number_input('Enter number of default months', min_value=-1 , max_value =  9)