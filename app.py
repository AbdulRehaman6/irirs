# Streamlit app for ML model training and prediction
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# App title
st.title("🌸 Iris Flower Classifier 🌸")
st.write("Enter the feature values to predict the Iris flower species.")

# Load the Iris dataset
@st.cache_data
def load_model():
    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # Train model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    return model, feature_names, target_names

model, feature_names, target_names = load_model()

# Manual input for features
st.subheader("Enter Feature Values")
user_input = []
for feature in feature_names:
    value = st.number_input(f"Enter value for {feature} (cm):", min_value=0.0, step=0.1)
    user_input.append(value)

# Prediction button
if st.button("Predict"):
    # Make prediction
    prediction = model.predict([user_input])
    predicted_class = target_names[int(prediction[0])]
    st.success(f"The predicted Iris species is: **{predicted_class}**")
