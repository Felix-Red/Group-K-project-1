import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import joblib

classifier = joblib.load('classifierModel.plk')


app = dash.Dash(__name__, external_stylesheets=['style.css'])

server = app.server
picture = "./belgium.png"

app.layout = html.Div([
   html.Div([
        html.Img(src= "https://th.bing.com/th/id/OIP.agOZIZ1Hk_bhYVSgY-rNNAHaHa?w=167&h=180&c=7&r=0&o=5&dpr=1.3&pid=1.7", style={'height': '50px', 'margin-right': '10px'}),
        html.H1("Loan Approval Prediction", style={'display': 'inline-block', 'vertical-align': 'middle'}),
    ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'margin-bottom': '20px'}),
    html.Div([
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
        dcc.Dropdown(
            id='dependents',
            options=[
                {'label': '0', 'value': 0},
                {'label': '1', 'value': 1},
                {'label': '2', 'value': 2},
                {'label': '3+', 'value': 3}
            ],
            placeholder='Select Number of dependents'
        ),
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
        dcc.Input(
            type="range",
            min=0,
            max=9000,
            value=4500,
            id="applicant-income-input",
            style={'max-width': '600px', 'margin': 'auto', 'text-align': 'center'}
        ),
        html.Div(id='applicant-income-output'),

        dcc.Input(
            type="range",
            min=0,
            max=4000,
            value=2000,
            id="coapplicant-income-input",
            style={'max-width': '600px', 'margin': 'auto', 'text-align': 'center'}
        ),
        html.Div(id='coapplicant-income-output'),

        dcc.Input(
            type="range",
            min=0,
            max=700.0,
            value=350.0,
            id="loan-amount-input",
            style={'max-width': '600px', 'margin': 'auto', 'text-align': 'center'}
        ),
        html.Div(id='loan-amount-output'),

        dcc.Input(
            type="range",
            min=0,
            max=360,
            value=180,
            id="loan-term-input",
            style={'max-width': '600px', 'margin': 'auto', 'text-align': 'center'}
        ),
        html.Div(id='loan-term-output'),

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
    ], style={'max-width': '600px', 'margin': 'auto', 'text-align': 'center'}),

    html.Div(id='prediction-output', style={'text-align': 'center'})
])

@app.callback(
    Output('applicant-income-output', 'children'),
    [Input('applicant-income-input', 'value')]
)
def update_applicant_income_output(value):
    return f"Applicant Income: {value}"

@app.callback(
    Output('coapplicant-income-output', 'children'),
    [Input('coapplicant-income-input', 'value')]
)
def update_coapplicant_income_output(value):
    return f"Co-Applicant Income: {value}"

@app.callback(
    Output('loan-amount-output', 'children'),
    [Input('loan-amount-input', 'value')]
)
def update_loan_amount_output(value):
    return f"Loan Amount: {value}"

@app.callback(
    Output('loan-term-output', 'children'),
    [Input('loan-term-input', 'value')]
)
def update_loan_term_output(value):
    return f"Loan Term: {value}"

@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [Input('gender-dropdown', 'value'),
     Input('marital-status-dropdown', 'value'),
     Input('dependents', 'value'),
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
        prediction = classifier.predict([features])
        if prediction[0] == 1:
            return "Loan Approved"
        else:
            return "Loan Not Approved"

if __name__ == '__main__':
    app.run_server(debug=True)
