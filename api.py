from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Define a root route
@app.get("/")
async def read_root():
    return {"message": "Welcome to the API"}

# Define the input data schema
class EmployeeData(BaseModel):
    no_of_trainings: int
    age: int
    previous_year_rating: int
    length_of_service: int
    avg_training_score: int
    education: str
    KPIs_met: bool
    awards_won: bool

# Load model, scaler, and columns (the logic from your Streamlit code)
def load_model():
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

    X = df.drop(['is_promoted', 'employee_id', 'department', 'region', 'gender', 'recruitment_channel', 'KPIs_met', 'awards_won'], axis=1)
    y = df['is_promoted']

    X = pd.get_dummies(X, columns=['education'])
    X = X.dropna()
    y = y[X.index]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=200, max_depth=5, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler, X.columns

model, scaler, feature_names = load_model()

# Education level mapping
education_mapping = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}

# Define the prediction endpoint
@app.post("/predict")
def predict(data: EmployeeData):
    # Convert education level to a numerical value
    education_num = education_mapping.get(data.education, 1)

    # Combine input features into a numpy array
    input_data = pd.DataFrame({
        'no_of_trainings': [data.no_of_trainings],
        'age': [data.age],
        'previous_year_rating': [data.previous_year_rating],
        'length_of_service': [data.length_of_service],
        'avg_training_score': [data.avg_training_score],
        'education': [data.education],
        'total_score': [data.no_of_trainings * data.avg_training_score],
        'performance': [1 if data.KPIs_met or data.awards_won else 0]
    })

    input_data = pd.get_dummies(input_data, columns=['education'])
    for col in feature_names:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[feature_names]

    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)
    promotion_probability = model.predict_proba(scaled_input)[0][1]

    return {
        "prediction": "Promote the employee" if prediction[0] == 1 else "Do not promote the employee",
        "promotion_probability": promotion_probability
    }
