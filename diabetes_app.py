import pandas as pd
import streamlit as st
import numpy as np
import matplotlib as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import  RandomForestClassifier
from sklearn.metrics import accuracy_score

df= pd.read_csv("diabetes.csv")

st.title("Diabetes Prediction App by Mohammad Wasiq")
st.sidebar.header("Paitent Data")
st.subheader("Description Statistics of Data")
st.write(df.describe())

# Data Split
X= df.drop(['Outcome'], axis=1)
y= df.iloc[ :, -1]
X_train, X_test, y_train, y_test= train_test_split(X, y, train_size= 0.8, random_state=0)

# Function
def user_report():
    pregnancies= st.sidebar.slider("Pregnancies", 0, 17, 2)
    glucose= st.sidebar.slider("Glucose", 0, 199, 110)
    bp= st.sidebar.slider("BloodPressure", 0, 122, 80)
    sk= st.sidebar.slider("SkinThikness", 0, 99, 12)
    insulin= st.sidebar.slider("Insulin", 0, 846, 80)
    bmi= st.sidebar.slider("BMI", 0, 67, 5)
    dpf= st.sidebar.slider("DiabetesPedigreeFunction", 0.07, 2.42, 0.37)
    age= st.sidebar.slider("Age", 21, 81, 33)

    user_report_data= {
        "pregnancies" : pregnancies,
        "glucose" : glucose,
        "bp" : bp,
        "sk" : sk,
        "insulin" : insulin,
        "bmi" : bmi,
        "dpf" : dpf,
        "age" : age
    }
    report_data= pd.DataFrame(user_report_data, index=[0])
    return report_data

# Patient data
user_data= user_report()
st.subheader("Patient Data")
st.write("user_data")

# Model
rc= RandomForestClassifier()
rc.fit(X_train, y_train)
user_result= rc.predict(user_data)

# Data Visulaization
st.title("Visualization Patient Data")

# Color Function
if user_result[0]==0:
    color= 'blue'
else:
    color= 'red'

# Age VS Pregnancies
st.header("Pregnancies Count (Others vs Yours)")
fig_preg= plt.figure()
ax1= sns.scatterplot(x= 'Age', y= 'Pregnancies', data= df, hue= 'Outcome', palette= "Greens")
ax2= sns.scatterplot(x= user_result['age'], y= user_data['pregnancies'], s= 150, color= color)
plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 20, 2))
plt.title("0 - Healthy & 1 - Diabetoc")
st.pyplot(fig_preg)

# Output
st.header("Your Report: ")
output= ''
if user_result[0]==0:
    output= "You are Healthy"
    st.balloons()
else:
    output= "You are Diabetic So Don't Eat Sweet"
    st.warning("Sugar", "Sugar", "Sugar")
    st.title(output)
    st.subheader("Accuracy: ")
    st.subheader(str(accuracy_score(y_test, rc.predict(X_test))*100 +"%"))
