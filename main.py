
from models.LSTM_model import run_lstm
from models.KNN_model import run_knn
from models.RandomForest_model import run_random_forest
from models.LinearRegression_model import run_linear_regression

# class App:
#     def __init__(self):
#         gui=Gui()
#         models=Something()



def main(modelname):
    """
    Change this to use "strategy pattern"
    """
    if  modelname == 'Dash':
        run_dashboard()
    elif  modelname == 'LSTM':
        run_lstm()
    elif modelname=='KNN':
        run_knn()
    elif modelname=='RandomForest':
        run_random_forest()
    elif modelname=='Linear':
        run_linear_regression()


if __name__=='__main__':
    model='KNN' # options are LSTM, KNN, RandForest, Linear
    main(model)

#