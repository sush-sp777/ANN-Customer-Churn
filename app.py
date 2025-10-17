import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import pickle
import warnings,os

warnings.filterwarnings("ignore",category=FutureWarning)
warnings.filterwarnings("ignore",category=UserWarning)

#streamlit app
st.title('Customer Churn Prediction App')
st.write("Use this app to predict whether a customer will leave the bank or not.")

@st.cache_resource
def load_model():
    if os.path.exists("model.keras"):
        return tf.keras.models.load_model("model.keras", compile=False, safe_mode=False)
    elif os.path.exists("model.h5"):
        return tf.keras.models.load_model("model.h5", compile=False)
    else:
        st.error("Model file not found. Please upload model.keras or model.h5")
        st.stop()

@st.cache_resource
def load_pickle(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)

# Load everything once (cached)
model = load_model()
label_encoder_gender = load_pickle("label_encoder_gender.pkl")
one_hot_encoder = load_pickle("one_hot_encoder.pkl")
scaler = load_pickle("scaler.pkl")


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
if st.button("Predict Churn"):
    prediction=model.predict(input_scaled)
    prediction_probability=prediction[0][0]
    st.write(f'Churn Probability:{prediction_probability:.2f}')

    if prediction_probability >0.5:
        st.write('The customer is likely to churn')
    else:
        st.write('The customer is not likely to churn')

