import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
from dash.dependencies import Output, Input

from models.KNN_model import run_knn
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
from models.RandomForest_model import run_random_forest

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}



model_names=['KNN', 'LSTM', 'LinearRegression','RandomForest']
# print(df)
# fig = px.scatter(df, x="Dates", y="Predicted")



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
        id='xaxis-column',
        options=[{'label': i, 'value': i} for i in model_names],
        value='Fertility rate, total (births per woman)'
    ),
    html.Div(id='dd-output-container'),

    dcc.Graph(id='indicator-graphic')
])


@app.callback(
    Output('indicator-graphic', 'figure'),
    Input('xaxis-column', 'value'))

def update_graph(model_name,):
    if model_name=='KNN':
        fig = run_knn()
    elif model_name=='LinearRegression':
        fig = run_linear_regression()
    elif model_name=='LSTM':
        fig = run_lstm()
    elif model_name=='RandomForest':
        fig = run_random_forest()
    fig = tls.mpl_to_plotly(fig)
    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text']
    )
    # dff = df[df['Year'] == year_value]
    #
    # fig = px.scatter(x=dff[dff['Indicator Name'] == xaxis_column_name]['Value'],
    #                  y=dff[dff['Indicator Name'] == yaxis_column_name]['Value'],
    #                  hover_name=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'])
    #
    # fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
    #
    # fig.update_xaxes(title=xaxis_column_name,
    #                  type='linear' if xaxis_type == 'Linear' else 'log')
    #
    # fig.update_yaxes(title=yaxis_column_name,
    #                  type='linear' if yaxis_type == 'Linear' else 'log')

    return fig

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
