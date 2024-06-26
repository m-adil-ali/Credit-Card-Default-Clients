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
    st.write('---')
    st.write("### Default History of past six months")
    st.write('- Please indicate the number of months in which you experienced defaults over the past six months, using values from 1 to 9 for defaults and -1 for no defaults. Put the values in the first six entries below:')
    HIST_9 = st.number_input('Enter number of default months for last month', min_value=-1 , max_value =  9)
    HIST_8 = st.number_input('Enter number of default months for second last month', min_value=-1 , max_value =  9)
    HIST_7 = st.number_input('Enter number of default months for third last month', min_value=-1 , max_value =  9)
    HIST_6 = st.number_input('Enter number of default months for fourth last month', min_value=-1 , max_value =  9)
    HIST_5 = st.number_input('Enter number of default months for fifth last month', min_value=-1 , max_value =  9)
    HIST_4 = st.number_input('Enter number of default months for sixth last month', min_value=-1 , max_value =  9)
    st.write('---')
    st.write("### Bill History of past six months")
    st.write('- Please enter the amount of Bill you received over the past six months, you can enter wide range of positive and negative numbers for bill you received. Put the amounts in the first six entries below:')
    BILL_9 = st.number_input('Enter the amount of Bill you received for last month', step= 1)
    BILL_8 = st.number_input('Enter the amount of Bill you received for second last month', step= 1)
    BILL_7 = st.number_input('Enter the amount of Bill you received for third last month', step= 1)
    BILL_6 = st.number_input('Enter the amount of Bill you received for fourth last month', step= 1)
    BILL_5 = st.number_input('Enter the amount of Bill you received for fifth last month', step= 1)
    BILL_4 = st.number_input('Enter the amount of Bill you received for sixth last month', step= 1)
    st.write('---')
    st.write("### Payment History of past six months")
    st.write('- Please enter the amount of Bill you paid over the past six months, you can enter wide range of positive numbers for the bill you paid in a month and \'0\' for the month you paid nothing. Put the amounts in the first six entries below:')
    PAY_9 = st.number_input('Enter the amount of Bill you paid for last month',min_value= 0, step= 1)
    PAY_8 = st.number_input('Enter the amount of Bill you paid for second last month', min_value= 0, step= 1)
    PAY_7 = st.number_input('Enter the amount of Bill you paid for third last month', min_value= 0, step= 1)
    PAY_6 = st.number_input('Enter the amount of Bill you paid for fourth last month', min_value= 0, step= 1)
    PAY_5 = st.number_input('Enter the amount of Bill you paid for fifth last month', min_value= 0, step= 1)
    PAY_4 = st.number_input('Enter the amount of Bill you paid for sixth last month', min_value= 0, step= 1)
    
    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    user_input = [LIMIT_BAL, HIST_9, HIST_8, HIST_7, HIST_6, HIST_5, HIST_4, BILL_9, BILL_8, BILL_7, BILL_6, BILL_5, BILL_4, 
                  PAY_9, PAY_8, PAY_7, PAY_6, PAY_5, PAY_4]
    
    #st.write("Before reshaping: ", user_input)
    
    # Convert the list to a 2D array with one row
    user_input = np.array(user_input).reshape(1,-1) 
    
    #st.write("After reshaping to 1D: ", user_input.shape, user_input)
    
    # Applying the same standard scaler transformation to the user input data through which we have fit.transformed X_train during model training.
    full_input_scaled = scaler.transform(user_input)
    
    # Predicting the result.
    prediction = model.predict(full_input_scaled)
    st.write('---')
    st.write('### Predictions based on your given input data:')
    if prediction == 0:
        st.write("You are not likely to default on your credit card next month.")
    else:
        st.write("You are likely to default on your credit card next month.")
     
    prediction_proba = model.predict_proba(full_input_scaled)
    
    st.subheader('Prediction Probability')
    st.write('1 means likelihood to default and 0 means likelihood to not default')
    st.write(prediction_proba)