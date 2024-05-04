# Hi, I'm Anil Kumar! 👋

## Diabetes Prediction
This Streamlit app predicts the likelihood of diabetes based on user input. It utilizes a Desicion Tree Classifier model trained on a diabetes dataset to make predictions.

## Features
- Input fields for user data including pregnancies, glucose level, blood pressure, skin thickness, insulin level, BMI, diabetes pedigree function, and age.
- Predicts whether the user has diabetes or not.
- Displays appropriate messages based on the prediction result.

## User Inputs
1. **Pregnancies**: 
    - The number of pregnancies the user has had.
    - This can be an important factor in predicting the risk of diabetes, as pregnancy-related changes in hormone levels can affect insulin sensitivity.
2. **Glucose Level**: 
    - The user's current glucose (blood sugar) level.
    - Elevated glucose levels are a key indicator of diabetes, as diabetes is characterized by high blood sugar levels.
3. **Blood Pressure**: 
    - The user's blood pressure reading.
    - High blood pressure can be associated with diabetes and its complications, making it relevant for predicting diabetes risk.
4. **Skin Thickness**: 
    - The thickness of the user's skin (measured in millimeters).
    - Skin thickness can be related to metabolic factors and may provide insights into diabetes risk.
5. **Insulin Level**: 
    - The user's insulin level (measured in mU/L or microunits per liter).
    - Insulin plays a crucial role in glucose metabolism, and abnormal insulin levels can be indicative of diabetes.
6. **BMI (Body Mass Index)**: 
    - The user's BMI, calculated based on their weight and height.
    - BMI is a common indicator of overall health and can be associated with diabetes risk, especially in cases of obesity.
7. **Diabetes Pedigree Function**: 
    - A numerical value representing the diabetes pedigree function.
    - This function estimates diabetes heredity and is based on family history and genetic factors.
8. **Age**: 
    - The user's age.
    - Age is a significant risk factor for diabetes, as the likelihood of developing diabetes increases with age.

## How to Use
1. Install the required libraries.
```bash
pip install -r requirements.txt
```
2. Run the Streamlit app.
```bash
stremalit run app.py 
```
If the above command does not works then try the below command
```bash
python -m stremalit run app.py 
```
3. Fill in the required fields with relevant data.
4. Click the "Make Prediction" button to see the prediction result.


## Dependencies
1. 'Streamlit': Streamlit is a Python library used for building interactive web applications. It provides simple APIs for creating user interfaces and visualizations.
```bash
import streamlit as st
```
2. `Pandas`: Pandas is a popular data manipulation and analysis library in Python. It is used for reading and processing data from various sources, such as CSV files.
```bash
import pandas as pd
```
3. `scikit-learn (sklearn)`: scikit-learn is a machine learning library in Python that provides various algorithms and tools for machine learning tasks such as classification, regression, clustering, and more.
```bash
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
```
4. `RandomForestClassifier`: RandomForestClassifier is a classification algorithm from the scikit-learn library. It is used to train a random forest model for predicting the outcome (diabetes or non-diabetes) based on input features.
5. `train_test_split`: train_test_split is a function from scikit-learn used for splitting the dataset into training and testing sets. It helps evaluate the model's performance on unseen data.

## Files included
- `Diabetes_Prediction.ipynb`: Contains the Jupyter Notebook for data analysis, preprocessing, model training, etc.
- `app.py`: Contains the Streamlit app code for the web interface and model prediction.
- `requirements.txt`: Lists the Python dependencies required for the project.
- `diabetes_dataset.csv`: The dataset file used for training and testing the model.
- `README.md`: Documentation file explaining the project, installation steps, usage instructions, etc.

## Random Forest Classifier
1. `Ensemble Learning`: Random Forest is an ensemble learning method that combines multiple decision trees to make predictions. Each decision tree in the forest is trained independently on random subsets of the data (bootstrap samples) and random subsets of features.
2. `Decision Trees`: A decision tree is a flowchart-like structure where each internal node represents a feature or attribute, each branch represents a decision based on that feature, and each leaf node represents the outcome or class label. Decision trees are used for both classification and regression tasks.
3. `Random Forest Algorithm`:
    - Random Forest builds multiple decision trees during training.
    - It randomly selects a subset of features at each split in each tree, reducing overfitting and improving generalization.
    - During prediction, each tree in the forest independently predicts the class label, and the final prediction is determined by majority voting (classification) or averaging (regression) across all trees.


## Model Training
1. Dataset: The model is trained on a diabetes dataset that includes features such as pregnancies, glucose level, blood pressure, skin thickness, insulin level, BMI, diabetes pedigree function, and age. The target variable is the diabetes outcome (0 for non-diabetes, 1 for diabetes).
2. Training Process:
    - The dataset is split into training and testing sets using the `train_test_split` function.
    - The Random Forest Classifier model is initialized with hyperparameters such as the number of trees (n_estimators) and random state.
    - The model is trained using the training data (X_train, y_train) to learn patterns and relationships between input features and the target variable.

## Model Evaluation
1. Testing Set: After training, the model's performance is evaluated on a separate testing set (X_test, y_test) to assess its accuracy and generalization ability.
2. Prediction: During prediction, the user inputs (e.g., pregnancies, glucose level, etc.) are collected and fed into the trained model. The model uses these inputs to predict the likelihood of diabetes (0 for non-diabetes, 1 for diabetes) based on the learned patterns.
3. Output: The model's prediction is displayed to the user through the Streamlit app, along with appropriate messages indicating whether the user is likely to have diabetes or not.

## Preview
[Checkout Here]()

## 🔗 Follow us
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/anilkumarkonathala/)

## Feedback
If you have any feedback, please reach out to us at konathalaanilkumar143@gmail.com
