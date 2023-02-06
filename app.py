import numpy as np
import pickle
import streamlit as st

#loading the saved model

loaded_model = pickle.load(open('diabetesmodel.pkl', 'rb'))

#creating a function for prediction

def diabetes_prediction(input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)

    if (prediction[0] == 0):
      return('The person is not diabetic.')
    else:
      return('The person is diabetic.')
  
def main():
    #give a title
    st.title('Diabetes Prediction Web App by Mohammad Wasiq')
        						    
    #getting the input from the user
    
    Pregnancies = st.text_input('No. of Pregnancies')
    Glucose = st.text_input('Blood Glucose Level')
    BloodPressure = st.text_input('Blood Pressure')
    SkinThickness = st.text_input('Skin Thickness')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    Age = st.text_input('Age of the Person')
    
    #code for prediction
    diagnosis = ''
    
    #creating a button for prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure,SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age ])
    
    st.success(diagnosis)
            
if __name__ == '__main__':
    main()
    
 
