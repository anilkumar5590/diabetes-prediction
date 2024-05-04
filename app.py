import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Set Streamlit page configuration
st.set_page_config(
    page_title="Diabetes Prediction ",
)

# Load the diabetes dataset
diabetes_df = pd.read_csv('diabetes_dataset.csv')

# Title of the Streamlit app
st.title('Diabetes Prediction')

# Input fields for user data
pregnancies = st.number_input('Pregnancies', value=None, placeholder='Enter no. of pregnancies ...', step=1)
glucose = st.number_input('Glucose', value=None, placeholder='Enter Glucose value ...', step=1)
bp = st.number_input('Blood Pressure', value=None, placeholder='Enter blood pressure value ... ', step=1)
skinthickness = st.number_input('Skin Thickness', value=None, placeholder='Enter skin thickness value ... ', step=1)
insulin = st.number_input('Insulin', value=None, placeholder='Enter insulin value ...', step=1)
bmi = st.number_input('BMI', value=None, placeholder='Enter body mass index(BMI) value ...', step=0.1)
diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', value=None, placeholder='Enter diabetes pedigree function value ...', step=0.001)
age = st.number_input('Age', value=None, placeholder='Enter your age ...', step=1)

# Define features (X) and target variable (y) for model training
y = diabetes_df['Outcome']
X = diabetes_df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                 'BMI', 'DiabetesPedigreeFunction', 'Age']]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# Create a Random Forest Classifier model
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model using the training data
random_forest_model.fit(X_train, y_train)

# Gather user input
user_input = [pregnancies, glucose, bp, skinthickness, insulin, bmi, diabetes_pedigree_function, age]

# Check if 'Make Prediction' button is clicked
if st.button('Make Prediction'):
    # Check if any input field is empty
    flag = 0
    for i in user_input:
        if i is None:
            flag = 1
            break
    if flag:
        # Display error message if any input field is empty
        st.error('⚠️ Please fill all the above fields ')
    else:
        # Make prediction using the Random Forest model
        prediction_value = random_forest_model.predict([user_input])
        if prediction_value == 1:
            # Display an error message if the prediction is positive for diabetes
            st.error('You have Diabetes. Please consult a doctor immediately 🧑🏻‍⚕️')
            # Uncomment the line below to display an image related to diabetes
            # st.image('diabetes.png', caption='Diabetes Image')
        elif prediction_value == 0:
            # Display a success message if the prediction is negative for diabetes
            st.success("You don't have diabetes. Feel Free!! ")

# Footer content
footer_html = """
<hr>
<div style="bottom: 0;  color: green; text-align: center;">
    <p style="font-weight: bold; ">Developed by Anil Kumar</p>
</div>
"""

# Display the footer using markdown
st.markdown(footer_html, unsafe_allow_html=True)
