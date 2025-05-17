import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

st.title("Salary Prediction Dashboard")

# Load or train model
try:
    model = pickle.load(open("rf_model.pkl", "rb"))
    le_gender = pickle.load(open("le_gender.pkl", "rb"))
    le_job = pickle.load(open("le_job.pkl", "rb"))
except:
    df = pd.read_csv("Salary_Data.csv")

    # Encode Gender and Job Title
    le_gender = LabelEncoder()
    le_job = LabelEncoder()
    df['Gender'] = le_gender.fit_transform(df['Gender'])
    df['Job Title'] = le_job.fit_transform(df['Job Title'])

    # Manually encode Education Level
    education_map = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
    df['Education Level'] = df['Education Level'].map(education_map)

    # Prepare training data
    X = df[['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']]
    y = df['Salary']

    # Train and save model
    model = RandomForestRegressor()
    model.fit(X, y)
    pickle.dump(model, open("rf_model.pkl", "wb"))
    pickle.dump(le_gender, open("le_gender.pkl", "wb"))
    pickle.dump(le_job, open("le_job.pkl", "wb"))

# Load for user input
salary_df = pd.read_csv("Salary.csv")  # For dropdown options + visualization

# User inputs
age = st.slider("Age", 18, 70, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
job_title = st.selectbox("Job Title", salary_df['Job Title'].unique())
experience = st.slider("Years of Experience", 0, 40, 5)

# Encode inputs
gender_encoded = le_gender.transform([gender])[0]
education_map = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
education_encoded = education_map[education]
job_encoded = le_job.transform([job_title])[0]

# Predict
input_data = [[age, gender_encoded, education_encoded, job_encoded, experience]]
if st.button("Predict Salary"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Predicted Salary: ${prediction:,.2f}")

# Visualization
fig = px.histogram(salary_df, x="Salary", title="Salary Distribution", nbins=30, color_discrete_sequence=["#636EFA"])
st.plotly_chart(fig)
