import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import pickle
import warnings

##Load the trained model
model=tf.keras.models.load_model('model.keras',safe_mode=False,compile=False)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('one_hot_encoder.pkl','rb') as file:
    one_hot_encoder=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

#streamlit app
st.title('Customer Churn Prediction App')
st.write("Use this app to predict whether a customer will leave the bank or not.")

#user input
geography=st.selectbox('Geography',one_hot_encoder.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number_of_products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])

#prepare the input data

input=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary],   
})

#one-hot encode Geography
geo_encoded=one_hot_encoder.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=one_hot_encoder.get_feature_names_out(['Geography']))

#combine input data with geography encoded column
input_data=pd.concat([input.reset_index(drop=True),geo_encoded_df],axis=1)

#scale the input data
input_scaled=scaler.transform(input_data)

#predict churn
prediction=model.predict(input_scaled)
prediction_probability=prediction[0][0]
st.write(f'Churn Probability:{prediction_probability:.2f}')

if prediction_probability >0.5:
    st.write('The customer is likely to churn')
else:
    st.write('The customer is not likely to churn')


warnings.filterwarnings("ignore",category=FutureWarning)
warnings.filterwarnings("ignore",category=UserWarning)


