import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
from dash.dependencies import Output, Input

from models.Ensemble_model import run_ensemble
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
from threading import Thread
import concurrent.futures

import pandas as pd

from models.LinearRegression_model import run_linear_regression
from models.RandomForest_model import run_random_forest


def get_models_hashmap():
    #store the models using hashmap
    print('#################################################')
    que = []

    knn_thread = Thread(target=que.append(("KNN",run_knn())), args=())
    knn_thread.start()

    lstm_thread = Thread(target=que.append(("LSTM",run_lstm())), args=())
    lstm_thread.start()

    linear_thread = Thread(target=que.append(("LinearRegression",run_linear_regression())), args=())
    linear_thread.start()

    random_forest_thread = Thread(target=que.append(("RandomForest",run_random_forest())), args=())
    random_forest_thread.start()

    ensemble = Thread(target=que.append(("Ensemble",run_ensemble())), args=())
    ensemble.start()

    knn_thread.join()
    lstm_thread.join()
    linear_thread.join()
    random_forest_thread.join()
    ensemble.join()

    return dict(que)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

model_names=['KNN', 'LSTM', 'LinearRegression','RandomForest','Ensemble']
# models_hashmap = get_models_hashmap()

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Predicting The Price Of Bitcoin',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    html.Div(children='Choose a Machine Learning Model', style={
        'textAlign': 'center',
        'color': colors['text']
    }),

    dcc.Dropdown(
        id='xaxis-column',
        options=[{'label': i, 'value': i} for i in model_names],
        value='KNN'
    ),
    html.Div(id='dd-output-container'),

    dcc.Graph(id='indicator-graphic')
])


@app.callback(
    Output('indicator-graphic', 'figure'),
    Input('xaxis-column', 'value'))

def update_graph(model_name,):
    # fig = models_hashmap[model_name]
    if model_name=='KNN':
        fig=run_knn()
    elif model_name=='RandomForest':
        fig=run_random_forest()
    elif model_name=='LinearRegression':
        fig=run_linear_regression()
    elif model_name=='LSTM':
        fig=run_lstm()
    elif model_name=='Ensemble':
        fig=run_ensemble()
    fig = tls.mpl_to_plotly(fig)
    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text']
    )
    return fig

if __name__=='__main__':
    app.run_server(debug=True)

    # show the predicted key and actual key on the side of the dashboard
    # change to the correct colors
    #improve ensemble model