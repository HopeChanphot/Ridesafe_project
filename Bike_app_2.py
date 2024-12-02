# Import necessary libraries
import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime, timedelta
import os

# Initialize Dash app with suppressed callback exceptions
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "RideSafe Dashboard"

# Load the trained models and scaler
model_bike_next_day_rf = joblib.load(r"C:\Users\user\Downloads\Model_bike\model_next_day_rf_optimized.pkl")
scaler = joblib.load(r"C:\Users\user\Downloads\Model_bike\scaler.pkl")  # Load the saved scaler

# Load the features list
features_bike = joblib.load(r"C:\Users\user\Downloads\Model_bike\features_list.pkl")

# App layout with UX/UI improvements
app.layout = html.Div([
    html.Div(
        style={
            "background-color": "#8a2be2",  # Violet color for header
            "padding": "15px", 
            "color": "white", 
            "textAlign": "center",
            "fontSize": "28px",
            "fontWeight": "bold",
            "boxShadow": "0px 4px 6px rgba(0, 0, 0, 0.1)"
        },
        children=[
           html.Img(src="/assets/RideSafe_logo.png", style={"height": "50px", "marginRight": "10px"}),
           "RideSafe Dashboard"
        ]
    ),
    dcc.Tabs(id="tabs", value="user_site", children=[
        dcc.Tab(
            label="User Site",
            value="user_site",
            style={"background-color": "#f5f5f5", "color": "#8a2be2"},
            selected_style={"background-color": "#8a2be2", "color": "white"}
        ),
        dcc.Tab(
            label="Seller Site",
            value="seller_site",
            style={"background-color": "#f5f5f5", "color": "#8a2be2"},
            selected_style={"background-color": "#8a2be2", "color": "white"}
        ),
    ]),
    html.Div(id="tabs-content", style={"padding": "20px", "fontFamily": "Arial, sans-serif", "background-color": "#e6e6fa"})  # Light purple background
])

# Callback to render tab content dynamically
@app.callback(Output("tabs-content", "children"), [Input("tabs", "value")])
def render_tab_content(tab):
    if tab == "user_site":
        return html.Div([
            html.H1("Air Quality Prediction Dashboard", style={"color": "#8a2be2", "textAlign": "center"}),
            html.P("Get insights into air quality for your bike-sharing rides.", style={"textAlign": "center"}),
            html.Div([
                html.Button("\ud83c\udf1f Predict Air Quality for Next Day", id="predict-next-day-air", n_clicks=0,
                            style={
                                "background-color": "#8a2be2", "color": "white",
                                "border": "none", "padding": "10px 20px",
                                "borderRadius": "5px", "cursor": "pointer", "fontSize": "16px"
                            }),
                html.Div(id="next-day-air-output", style={"margin-top": "20px", "textAlign": "center", "fontSize": "18px"}),
            ], style={"textAlign": "center", "marginBottom": "20px"}),
        ])
    elif tab == "seller_site":
        return html.Div([
            html.H1("Bike Availability Prediction", style={"color": "#8a2be2", "textAlign": "center"}),
            html.P("Predict bike availability for better planning.", style={"textAlign": "center"}),

            html.Div([
                html.Label('Temperature (t1):'),
                dcc.Input(id='input-t1', type='number', value=30, style={'marginBottom': '10px'}),
                html.Label('Temperature (t2):'),
                dcc.Input(id='input-t2', type='number', value=30, style={'marginBottom': '10px'}),
                html.Label('Humidity:'),
                dcc.Input(id='input-hum', type='number', value=70, style={'marginBottom': '10px'}),

                html.Button("\ud83c\udf1f Predict Bike Availability for Next Day", id="predict-bike-next-day", n_clicks=0,
                            style={
                                "background-color": "#3498db", "color": "white",
                                "border": "none", "padding": "10px 20px",
                                "borderRadius": "5px", "cursor": "pointer", "fontSize": "16px",
                                "marginTop": "20px"  # Button placed under all other input fields
                            }),
                html.Div(id="bike-next-day-output", style={"margin-top": "20px", "textAlign": "center", "fontSize": "18px"}),
            ], style={"textAlign": "center"}),
        ])

# Callback for bike availability prediction
@app.callback(
    Output("bike-next-day-output", "children"),
    Input("predict-bike-next-day", "n_clicks"),
    [State('input-t1', 'value'),
     State('input-t2', 'value'),
     State('input-hum', 'value')]
)
def predict_bike_availability_next_day(n_clicks, t1, t2, hum):
    if n_clicks > 0:
        # Prepare the input data for prediction
        input_data = pd.DataFrame([{
            't1': t1,
            't2': t2,
            'hum': hum
        }])

        # Ensure input_data has the correct features and in the correct order
        input_data = input_data[features_bike]

        # Debugging output to check input data before scaling
        print("Input Data Before Scaling:")
        print(input_data)

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Debugging output to check scaled input data
        print("Input Data After Scaling:")
        print(input_data_scaled)

        # Predict using the RandomForestRegressor model
        y_pred = model_bike_next_day_rf.predict(input_data_scaled)
        
        # Debugging output to check raw predictions
        print("Raw Prediction:")
        print(y_pred)

        y_pred = np.expm1(y_pred)  # Inverse log transformation
        
        # Debugging output to check transformed predictions
        print("Transformed Prediction:")
        print(y_pred)

        y_pred_rounded = np.round(y_pred).astype(int)  # Round predictions

        return f"Predicted Bike Availability for Tomorrow ({(datetime.now().date() + timedelta(days=1)).strftime('%A, %B %d, %Y')}): {y_pred_rounded[0]}"
    return ""

# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True, port=8051)



















