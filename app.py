import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import pickle


#Load the trained model
model = tf.keras.models.load_model('model.h5')


##Load the encoder and scaler file
with open('onehot_encoder_geo.pkl','rb') as f:
    label_encoder_geo = pickle.load(f)


with open('label_encoder_gender.pkl','rb') as f:
    label_encoder_gender = pickle.load(f)

with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)    


## Streamlit app

st.title("Customer Churn prediction")

# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

#One hot encode the geography
geo_encoded = label_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns = label_encoder_geo.get_feature_names_out['Geography'])


#Combine one hot encoded columns with input df
input_df =  pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)


#Scaled the input data
input_data_scaled = scaler.transform(input_df)


#Predict churn
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

if prediction_prob >0.5:
    st.write(f"The customer is likely to churn with a probability of {prediction_prob:.2f}")
else:
    st.write(f"The customer is not likely to churn with a probability of {prediction_prob:.2f}")    