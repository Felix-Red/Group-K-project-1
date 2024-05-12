import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import skimpy as sk
from category_encoders import OneHotEncoder
import matplotlib.pyplot as plt
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

loan_df = pd.read_csv("./data/raw_data.csv")
loan_val = pd.read_csv("./data/validation.csv")
#loan_val.head()
#loan_df.head()


#print(f'The dataframe has {loan_df.shape[0]} rows and {loan_df.shape[1]} columns.')

loan_df['LoanAmount'].fillna(loan_df['LoanAmount'].mean(), inplace=True)
loan_df['Credit_History'].fillna(loan_df['Credit_History'].mode()[0], inplace=True)
loan_df['Married'].fillna(loan_df['Married'].mode()[0], inplace=True)
loan_df['Gender'].fillna(loan_df['Gender'].mode()[0], inplace=True)
loan_df['Dependents'].fillna(loan_df['Dependents'].mode()[0], inplace=True)
loan_df['Self_Employed'].fillna(loan_df['Self_Employed'].mode()[0], inplace=True)
loan_df['Loan_Amount_Term'].fillna(loan_df['Loan_Amount_Term'].mode()[0], inplace=True)
#loan_df.info()

# Dropping the loan id column
loan_df = loan_df.drop(columns=['Loan_ID'], inplace=False)
#loan_df.head()

loan_df.replace({"Married": {"Yes":1, "No":0}, "Gender":{"Male":1, "Female": 0}, "Dependents":{"3+":3}, "Education":{"Graduate":1,"Not Graduate":0}, "Property_Area": {"Urban":2, "Semiurban":1, "Rural":0}, "Loan_Status": {"Y":1, "N":0}, "Self_Employed": {"Yes":1, "No":0}}, inplace=True)

X = loan_df.drop(columns = ["Loan_Status"], axis =1)
Y = loan_df["Loan_Status"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
#print(X.shape, X_train.shape, X_test.shape)

kf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
lg_model = LogisticRegression(random_state=1 ,max_iter=1000) 
lg_model.fit(X_train, Y_train)

pred_cv = lg_model.predict(X_train)
accuracy_score(Y_train,pred_cv)



# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of your app
app.layout = html.Div([
    html.H1("Loan Approval Prediction"),
    dcc.Dropdown(
        id='gender-dropdown',
        options=[
            {'label': 'Male', 'value': 1},
            {'label': 'Female', 'value': 0}
        ],
        placeholder="Select Gender"
    ),
    dcc.Dropdown(
        id='marital-status-dropdown',
        options=[
            {'label': 'Married', 'value': 1},
            {'label': 'Single', 'value': 0},
            {'label': 'Divorced', 'value': 2}
        ],
        placeholder="Select Marital Status"
    ),
    dcc.Input(id='dependents-input', type='number', placeholder='Enter Number of Dependents'),
    dcc.Dropdown(
        id='education-dropdown',
        options=[
            {'label': 'Graduated', 'value': 1},
            {'label': 'Not Graduated', 'value': 0}
        ],
        placeholder="Select Education Status"
    ),
    dcc.Dropdown(
        id='self-employed-dropdown',
        options=[
            {'label': 'Yes', 'value': 1},
            {'label': 'No', 'value': 0}
        ],
        placeholder="Select Self Employment Status"
    ),
    dcc.Input(id='applicant-income-input', type='number', placeholder='Enter Applicant Income'),
    dcc.Input(id='coapplicant-income-input', type='number', placeholder='Enter Co-Applicant Income'),
    dcc.Input(id='loan-amount-input', type='number', placeholder='Enter Loan Amount'),
    dcc.Input(id='loan-term-input', type='number', placeholder='Enter Loan Term'),
    dcc.Dropdown(
        id='credit-history-dropdown',
        options=[
            {'label': 'Good', 'value': 1.0},
            {'label': 'Bad', 'value': 0.0}
        ],
        placeholder="Select Credit History"
    ),
    dcc.Dropdown(
        id='residential-area-dropdown',
        options=[
            {'label': 'Urban', 'value': 2},
            {'label': 'Rural', 'value': 0},
            {'label': 'Semi-Urban', 'value': 1}
        ],
        placeholder="Select Residential Area"
    ),
    html.Button('Predict', id='predict-button', n_clicks=0),

    # Display the prediction result
    html.Div(id='prediction-output')
])

# Define callback to predict loan approval
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [Input('gender-dropdown', 'value'),
     Input('marital-status-dropdown', 'value'),
     Input('dependents-input', 'value'),
     Input('education-dropdown', 'value'),
     Input('self-employed-dropdown', 'value'),
     Input('applicant-income-input', 'value'),
     Input('coapplicant-income-input', 'value'),
     Input('loan-amount-input', 'value'),
     Input('loan-term-input', 'value'),
     Input('credit-history-dropdown', 'value'),
     Input('residential-area-dropdown', 'value')]
)
def predict_loan_approval(n_clicks, gender, marital_status, dependents, education, self_employed, applicant_income, coapplicant_income, loan_amount, loan_term, credit_history, residential_area):
    if n_clicks > 0:
        # Assuming you have preprocessed your input data appropriately
        # Concatenate all features into a single list
        features = [gender, marital_status, dependents, education, self_employed, applicant_income, coapplicant_income, loan_amount, loan_term, credit_history, residential_area]
        # Use your model to predict
        prediction = lg_model.predict([features])
        if prediction[0] == 1:
            return "Loan Approved"
        else:
            return "Loan Not Approved"
        


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
  

