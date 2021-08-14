import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

from models.LSTM_model import run_lstm


# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.tools as tls

import pandas as pd

from models.LinearRegression_model import run_linear_regression

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

fig = run_linear_regression()
fig = tls.mpl_to_plotly(fig)

# print(df)
# fig = px.scatter(df, x="Dates", y="Predicted")

fig.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Hello Dash',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    html.Div(children='Dash: A web application framework for Python.', style={
        'textAlign': 'center',
        'color': colors['text']
    }),

    dcc.Dropdown(
        id='demo-dropdown',
        options=[
            {'label': 'New York City', 'value': 'NYC'},
            {'label': 'Montreal', 'value': 'MTL'},
            {'label': 'San Francisco', 'value': 'SF'}
        ],
        placeholder="Select a model",
        value='NYC'
    ),
    html.Div(id='dd-output-container'),


    dcc.Graph(
        id='example-graph-2',
        figure=fig
    )
])

app.run_server(debug=True)

# def run_dashboard():
#     app = dash.Dash()
#     app.layout = html.Div(children =[]
#         html.H1('Bitcoin Prediction models'),
#
#         html.Div(children= '')
#         dcc.Dropdown(id= 'choose model'),
#         options=[
#         {'Linear': 'RandomForest', 'LSTM': 'KNN'},
#         html.Div(id="output-graph")
#     ],
#
#         dcc.Input(id="input", value='', type='text'),
#         html.Div(id="output-graph")
#     })
#
# # setting up the callback function and how the user will interact with the app
#
#     @app.callback(
#         Output(component_id="output-graph", component_property='children',
#         Input(component_id="my-input", component_property="value"  )
#     )
#
# # function that updates value
#     def update_value(input_value):
#         return 'Output: {}'.format(input_value)
# #   https: // dash.plotly.com / dash - core - components
