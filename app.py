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

# Load the trained models and scalers
model_max_hour_rf = joblib.load(os.path.join("models", "model_max_hour_rf_optimized5.pkl"))
scaler_bike = joblib.load(os.path.join("models", "scaler5.pkl"))

model_air_quality = joblib.load(os.path.join("models", "air_quality_model.pkl"))
scaler_air = joblib.load(os.path.join("models", "scalerq.pkl"))

# Load the features list
features_bike = joblib.load(os.path.join("models", "features_list5.pkl"))
features_air = ['Nitric Oxide', 'Nitrogen Dioxide', 'Ozone', 'hour', 'dayofweek', 'month']

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
                html.Button("🌟 Predict Air Quality for Tomorrow", id="predict-next-day-air", n_clicks=0,
                            style={
                                "background-color": "#8a2be2", "color": "white",
                                "border": "none", "padding": "10px 20px",
                                "borderRadius": "5px", "cursor": "pointer", "fontSize": "16px"
                            }),
                # Display next day's date below the button
                html.Div(id="next-day-date-display", style={"margin-top": "10px", "fontSize": "18px", "color": "#8a2be2", "textAlign": "center"}),

                html.Div(id="next-day-air-output", style={"margin-top": "20px", "textAlign": "center", "fontSize": "18px"}),
            ], style={"textAlign": "center", "marginBottom": "20px"}),
        ])
    elif tab == "seller_site":
        return html.Div([
            html.H1("Bike Availability Prediction", style={"color": "#8a2be2", "textAlign": "center"}),
            html.P("Predict peak bike availability for better planning.", style={"textAlign": "center"}),

            # Inputs section
            html.Div([
                html.Label('Temperature (t1):', style={'display': 'block', 'marginBottom': '5px'}),
                dcc.Input(id='input-t1', type='number', value=30, style={'marginBottom': '20px', 'width': '150px'}),

                html.Label('PM2.5 Particulate:', style={'display': 'block', 'marginBottom': '5px'}),
                dcc.Input(id='input-pm25', type='number', value=15, style={'marginBottom': '20px', 'width': '150px'}),

                html.Label('Humidity:', style={'display': 'block', 'marginBottom': '5px'}),
                dcc.Input(id='input-hum', type='number', value=70, style={'marginBottom': '40px', 'width': '150px'}),  # More space below humidity input to place the button

                # Button section, ensuring it is directly below the inputs
                html.Div([
                    html.Button("🌟 Predict Bike Demand for Tomorrow", id="predict-bike-peak-hour", n_clicks=0,
                                style={
                                    "background-color": "#3498db", "color": "white",
                                    "border": "none", "padding": "10px 20px",
                                    "borderRadius": "5px", "cursor": "pointer", "fontSize": "16px",
                                    "marginTop": "20px"  # Add spacing between the inputs and the button
                                }),
                    html.Div(id="bike-peak-hour-output", style={"margin-top": "20px", "textAlign": "center", "fontSize": "18px"})
                ])
            ], style={"textAlign": "center", "marginBottom": "20px"})
        ])

# Callback for air quality prediction, including displaying next day's date
@app.callback(
    [Output("next-day-air-output", "children"),
     Output("next-day-date-display", "children")],
    Input("predict-next-day-air", "n_clicks")
)
def predict_air_quality_next_day(n_clicks):
    if n_clicks > 0:
        next_day = (datetime.now() + timedelta(days=1)).strftime('%A, %B %d, %Y')
        # Create dummy input data for air quality prediction
        predictions = []
        for hour in range(24):
            input_data = pd.DataFrame([{
                'Nitric Oxide': 20,
                'Nitrogen Dioxide': 30,
                'Ozone': 40,
                'hour': hour,
                'dayofweek': (datetime.now().weekday() + 1) % 7,
                'month': datetime.now().month
            }])

            # Ensure input_data has the correct features and in the correct order
            input_data = input_data[features_air]

            # Scale the input data
            input_data_scaled = scaler_air.transform(input_data)

            # Predict using the ExtraTreesRegressor model
            y_pred = model_air_quality.predict(input_data_scaled)

            y_pred_rounded = np.round(y_pred).astype(int)
            predictions.append({'Hour': hour, 'Predicted AQI': y_pred_rounded[0], 'Category': categorize_air_quality('PM2.5', y_pred_rounded[0])})

        # Create DataFrame for table output
        predictions_df = pd.DataFrame(predictions)

        # Render table with centered text
        air_quality_output = dash_table.DataTable(
            data=predictions_df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in predictions_df.columns],
            style_table={'margin': '20px auto', 'overflowX': 'auto'},
            style_header={
                'backgroundColor': '#8a2be2',
                'fontWeight': 'bold',
                'color': 'white',
                'textAlign': 'center'
            },
            style_data={
                'backgroundColor': '#f5f5f5',
                'color': '#333',
                'textAlign': 'center'
            }
        )
        return air_quality_output, f"Predicted Air Quality for Tomorrow ({next_day})"
    return "", ""

# Function to categorize air quality
def categorize_air_quality(pollutant, value):
    bands = {
        "PM2.5": [(0, 11, 'Low'), (12, 23, 'Low'), (24, 35, 'Low'),
                  (36, 41, 'Moderate'), (42, 47, 'Moderate'), (48, 53, 'Moderate'),
                  (54, 58, 'High'), (59, 64, 'High'), (65, 70, 'High'), (71, float('inf'), 'Very High')]
    }
    for min_val, max_val, category in bands[pollutant]:
        if min_val <= value <= max_val:
            return category
    return "Unknown"

# Callback for bike peak hourly demand prediction
@app.callback(
    Output("bike-peak-hour-output", "children"),
    Input("predict-bike-peak-hour", "n_clicks"),
    [State('input-t1', 'value'),
     State('input-pm25', 'value'),
     State('input-hum', 'value')]
)
def predict_bike_peak_hour(n_clicks, t1, pm25, hum):
    if n_clicks > 0:
        # Prepare the input data for prediction
        input_data = pd.DataFrame([{
            't1': t1,
            'PM2.5 Particulate': pm25,
            'hum': hum,
            'hour': 12  # Default hour; we are predicting max hourly demand for the entire day
        }])

        # Ensure input_data has the correct features and in the correct order
        input_data = input_data[features_bike]

        # Scale the input data
        input_data_scaled = scaler_bike.transform(input_data)

        # Predict using the RandomForest model for max hourly demand
        y_pred = model_max_hour_rf.predict(input_data_scaled)

        # Apply inverse log transformation to the predictions
        y_pred = np.expm1(y_pred)

        # Round predictions and display result
        y_pred_rounded = np.round(y_pred).astype(int)

        return f" Predicted Bike Demand for Tomorrow ({(datetime.now().date() + timedelta(days=1)).strftime('%A, %B %d, %Y')}): {y_pred_rounded[0]} bikes"
    return ""


# Run the Dash app
#if __name__ == "__main__":
#    app.run_server(debug=True, host="0.0.0.0", port=8050)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Get the PORT from environment variable, default to 8080
    app.run_server(debug=True, host="0.0.0.0", port=port)