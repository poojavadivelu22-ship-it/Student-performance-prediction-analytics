import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Student Performance Prediction", layout="wide")

st.title("🎓 Student Performance Prediction System")

# Dataset
data = {
    'Attendance_Percentage':[92,65,88,55,95,70,45,85,78,50,90,60,82,48,96],
    'Study_Hours_Per_Day':[4,1,3,2,5,2,1,4,3,1,5,2,3,1,6],
    'Assignment_Score':[85,50,75,40,90,60,35,80,68,42,88,55,72,38,94],
    'Internal_Marks':[78,45,70,38,85,55,30,76,64,36,82,48,69,34,90],
    'Participation':['Yes','No','Yes','No','Yes','No','No','Yes','Yes','No','Yes','No','Yes','No','Yes'],
    'Internet_Access':['Yes','Yes','Yes','No','Yes','Yes','No','Yes','Yes','No','Yes','Yes','Yes','No','Yes'],
    'Previous_Sem_Marks':[80,40,72,35,88,58,28,79,66,32,84,46,71,30,91],
    'Final_Result':['Pass','Fail','Pass','Fail','Pass','Pass','Fail','Pass','Pass','Fail','Pass','Fail','Pass','Fail','Pass']
}

df = pd.DataFrame(data)

st.subheader("📊 Student Dataset")
st.dataframe(df)

# Encoding
le_part = LabelEncoder()
le_net = LabelEncoder()
le_result = LabelEncoder()

df['Participation'] = le_part.fit_transform(df['Participation'])
df['Internet_Access'] = le_net.fit_transform(df['Internet_Access'])
df['Final_Result'] = le_result.fit_transform(df['Final_Result'])

# Features and Target
X = df.drop('Final_Result', axis=1)
y = df['Final_Result']

# Model Training
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

st.subheader("🔍 Predict Student Result")

attendance = st.slider("Attendance Percentage", 0, 100, 75)
study_hours = st.slider("Study Hours Per Day", 0, 10, 3)
assignment = st.slider("Assignment Score", 0, 100, 70)
internal = st.slider("Internal Marks", 0, 100, 65)
participation = st.selectbox("Participation", ["Yes", "No"])
internet = st.selectbox("Internet Access", ["Yes", "No"])
previous_marks = st.slider("Previous Semester Marks", 0, 100, 70)

if st.button("Predict Result"):
    participation_val = le_part.transform([participation])[0]
    internet_val = le_net.transform([internet])[0]

    prediction = model.predict([[
        attendance,
        study_hours,
        assignment,
        internal,
        participation_val,
        internet_val,
        previous_marks
    ]])

    result = le_result.inverse_transform(prediction)[0]

    if result == "Pass":
        st.success(f"✅ Predicted Result: {result}")
    else:
        st.error(f"❌ Predicted Result: {result}")
