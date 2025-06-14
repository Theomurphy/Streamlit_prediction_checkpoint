import pandas as pd 
from sklearn.metrics import accuracy_score, classification_report 

import joblib
import streamlit as st 
import os

# Charger le modèle ML

model_path = os.path.join("model", "LR_model.pkl")

model = joblib.load(model_path)
# Load the trained model


def main():
    
    st.title('Prediction')
    
    country = st.selectbox("Select your country", ['Kenya', 'Rwanda', 'Tanzania', 'Uganda'])
    if country == 'Kenya':
        country =0
    elif country == 'Rwanda':
        country = 1
    elif country == 'Tanzania':
        country = 2
    else:
        country = 3
    
    year = st.selectbox('Select the year of the survey', [2018, 2016, 2017])
    
    location_type = st.radio('Type of location', ['Rural', 'Urban'])
    
    if location_type	 == 'Urban':
        location_type = 1
    else:
        location_type = 0
    
    cellphone_access = st.radio('Does the interviewee have access to a cellphone', ['Yes', 'No'])
    
    if cellphone_access	 == 'Yes':
        cellphone_access = 1
    else:
        cellphone_access = 0
    
    household_size = st.number_input('How many individuals live under the same roof :', 1, 21)
    
    age_of_respondent = st.number_input('Enter the age of the person being interviewed:', 16, 100)
    
    gender_of_respondent = st.radio('Select the gender of the interviewed person', ['Male', 'Female'])
    
    if gender_of_respondent	 == 'Male':
        gender_of_respondent = 1
    else:
        gender_of_respondent = 0
    
    relationship_with_head = st.selectbox('Select the Report of the interviewed person with the family holder', ['Spouse', 'Child', 'Parent', 'Head of Household', 'Other relative', 'Other non-relatives'])
    
    if relationship_with_head == 'Spouse':
        relationship_with_head = 5
    elif relationship_with_head == 'Head of Household':
        relationship_with_head = 1
    elif relationship_with_head == 'Other relative':
        relationship_with_head = 3
    elif relationship_with_head == 'Child':
        relationship_with_head = 0
    elif relationship_with_head == 'Parent':
        relationship_with_head = 4
    else:
        relationship_with_head = 2
    
    
    marital_status = st.selectbox('Select the marital status of the interviewee', ['Married/Living together', 'Widowed', 'Single/Never Married', 'Divorced/Seperated', 'Dont know'])
    
    if marital_status == 'Married/Living together':
        marital_status = 2
    elif marital_status == 'Widowed':
        marital_status = 4
    elif marital_status == 'Single/Never Married':
        marital_status = 3 
    elif marital_status == 'Divorced/Seperated':
        marital_status = 0 
    else:   
        marital_status = 1
    
    education_level = st.selectbox('Enter the Highest level of education the interviewed person', ['Secondary education', 'No formal education', 'Vocational/Specialised training', 'Primary education','Tertiary education', 'Other/Dont know/RTA'])
    
    if education_level == 'Secondary education':
        education_level = 3
    elif education_level == 'No formal education':
        education_level = 0
    elif education_level =='Vocational/Specialised training':
        education_level = 5
    elif education_level == 'Primary education':
        education_level = 2
    elif education_level == 'Tertiary education':
        education_level = 4
    else:
        education_level = 1
        
    job_lst = ['Dont Know/Refuse to answer', 'Farming and Fishing', 'Formally employed Government', 'Formally employed Private',  'Government Dependent' ,  'Informally employed', 'No Income',  'Other Income',  'Remittance Dependent', 'Self employed']
    
    job_type = st.selectbox('Type of job of the interviewee', job_lst)
    
    for i, job in enumerate(job_lst):
        if job_type == job:
            job_type = i


# Reconstructing the dataframe from the input data

    input_data = pd.DataFrame({'country':[country] , 'year':[year], 'location_type':[location_type], 'cellphone_access':[cellphone_access],'household_size':[household_size],
                               'age_of_respondent':[age_of_respondent], 'gender_of_respondent':[gender_of_respondent],'relationship_with_head':[relationship_with_head],
                               'marital_status':[marital_status], 'education_level':[education_level],'job_type':[job_type]} , index=[0])
    
    button = st.button('Make prediction')
    
    if button:
        predict_new = model.predict(input_data)
        
        if predict_new == 1:
            st.success('The interviewed person has a bank account')
        else:
            st.error('The interviewed person does not have a bank account')


main()
    