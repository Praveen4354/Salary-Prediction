import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.title("Salary Prediction Dashboard")

# Load or train model
try:
    model = pickle.load(open("rf_model.pkl", "rb"))
    le_gender = pickle.load(open("le_gender.pkl", "rb"))
    le_job = pickle.load(open("le_job.pkl", "rb"))
except:
    df = pd.read_csv("Salary_Data.csv")
    le_gender = LabelEncoder()
    le_job = LabelEncoder()
    df['Gender'] = le_gender.fit_transform(df['Gender'])
    df['Job Title'] = le_job.fit_transform(df['Job Title'])
    X = df[['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']]
    y = df['Salary']
    model = RandomForestRegressor()
    model.fit(X, y)
    pickle.dump(model, open("rf_model.pkl", "wb"))
    pickle.dump(le_gender, open("le_gender.pkl", "wb"))
    pickle.dump(le_job, open("le_job.pkl", "wb"))

# User inputs
age = st.slider("Age", 18, 70, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
job_title = st.selectbox("Job Title", pd.read_csv("Salary.csv")['Job Title'].unique())
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
    st.write(f"Predicted Salary: ${prediction:,.2f}")

# Visualization
import plotly.express as px
df = pd.read_csv("Salary.csv")
fig = px.histogram(df, x="Salary", title="Salary Distribution")
st.plotly_chart(fig)