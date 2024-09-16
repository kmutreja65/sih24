import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
import streamlit as st

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("employeePromotion.csv")
    cols = ['employee_id', 'department', 'region', 'education', 'gender',
            'recruitment_channel', 'no_of_trainings', 'age', 'previous_year_rating',
            'length_of_service', 'KPIs_met', 'awards_won',
            'avg_training_score', 'is_promoted']
    df.columns = cols
    
    df['previous_year_rating'] = df['previous_year_rating'].fillna(0)
    df['education'] = df['education'].fillna(df['education'].mode()[0])
    df['total_score'] = df['no_of_trainings'] * df['avg_training_score']
    df['performance'] = df[['KPIs_met', 'awards_won']].any(axis=1, skipna=False).astype(int)
    
    return df

# Train model
@st.cache_resource
def train_model(df):
    X = df.drop(['is_promoted', 'employee_id', 'department', 'region', 'gender', 'recruitment_channel', 'KPIs_met', 'awards_won'], axis=1)
    y = df['is_promoted']
    
    # Convert categorical variables to numeric
    X = pd.get_dummies(X, columns=['education'])
    
    # Ensure all columns are numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Drop any rows with NaN values
    X = X.dropna()
    y = y[X.index]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=200, max_depth=5, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X.columns

# Streamlit UI
st.set_page_config(page_title="Employee Promotion Predictor", page_icon="🚀", layout="wide")

st.title("🚀 Employee Promotion Predictor")

# Load data and train model
df = load_and_preprocess_data()
model, scaler, feature_names = train_model(df)

# Sidebar for user input
st.sidebar.header("📊 Employee Information")

# Use columns for better layout
col1, col2 = st.sidebar.columns(2)

with col1:
    no_of_trainings = st.number_input("Number of Trainings", 1, 10, 1)
    age = st.number_input("Age", 20, 60, 30)
    previous_year_rating = st.slider("Previous Year Rating", 0, 5, 3)

with col2:
    length_of_service = st.number_input("Length of Service (years)", 1, 20, 5)
    avg_training_score = st.number_input("Average Training Score", 0, 100, 70)

education = st.sidebar.selectbox("Education", df['education'].unique())
gender = st.sidebar.selectbox("Gender", df['gender'].unique())
department = st.sidebar.selectbox("Department", df['department'].unique())

KPIs_met = st.sidebar.checkbox("KPIs Met")
awards_won = st.sidebar.checkbox("Awards Won")

# Main content
st.header("🧑‍💼 Employee Profile")

# Use columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Number of Trainings", no_of_trainings)
    st.metric("Age", age)
    st.metric("Previous Year Rating", previous_year_rating)

with col2:
    st.metric("Length of Service", f"{length_of_service} years")
    st.metric("Avg Training Score", avg_training_score)
    st.metric("Education", education)

with col3:
    st.metric("Gender", gender)
    st.metric("Department", department)
    st.metric("Performance", "High" if KPIs_met or awards_won else "Standard")

# Calculate derived features
total_score = no_of_trainings * avg_training_score
performance = 1 if KPIs_met or awards_won else 0

# Create input dataframe
input_data = pd.DataFrame({
    'no_of_trainings': [no_of_trainings],
    'age': [age],
    'previous_year_rating': [previous_year_rating],
    'length_of_service': [length_of_service],
    'avg_training_score': [avg_training_score],
    'education': [education],
    'total_score': [total_score],
    'performance': [performance]
})

# Encode categorical variables
input_data = pd.get_dummies(input_data, columns=['education'])

# Ensure all columns from training are present
for col in feature_names:
    if col not in input_data.columns:
        input_data[col] = 0

# Reorder columns to match training data
input_data = input_data[feature_names]

# Scale input data
scaled_input = scaler.transform(input_data)

# Make prediction
prediction = model.predict(scaled_input)
promotion_probability = model.predict_proba(scaled_input)[0][1]

# Display prediction
st.header("🎯 Promotion Prediction")

col1, col2 = st.columns(2)

with col1:
    if prediction[0] == 1:
        st.success("This employee is likely to be promoted! 🎉")
    else:
        st.error("This employee is unlikely to be promoted at this time. 🤔")

with col2:
    st.progress(promotion_probability)
    st.write(f"Probability of promotion: {promotion_probability:.2%}")

# Recommendations
st.header("💡 Recommendations")

if prediction[0] == 0:
    st.write("Based on the model's prediction, here are some recommendations to improve promotion chances:")
    recommendations = [
        "Participate in more training programs to increase your skills and knowledge.",
        "Focus on improving your performance to meet KPIs and potentially win awards.",
        "Seek feedback from your manager on areas for improvement.",
        "Consider taking on additional responsibilities or projects to demonstrate leadership potential."
    ]
    for rec in recommendations:
        st.write(f"- {rec}")
else:
    st.write("Congratulations on your high promotion probability! Here are some tips to maintain your excellent performance:")
    tips = [
        "Continue your strong performance and meeting KPIs.",
        "Mentor junior employees to demonstrate leadership skills.",
        "Stay updated with industry trends and continue learning.",
        "Propose innovative ideas to further contribute to the company's success."
    ]
    for tip in tips:
        st.write(f"- {tip}")

# Disclaimer
st.caption("Note: This prediction is based on historical data and should be used as a guide only. Many factors contribute to promotion decisions.")