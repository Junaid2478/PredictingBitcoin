from models.Ensemble_model import run_ensemble
from models.LSTM_model import run_lstm
from models.KNN_model import run_knn
from models.RandomForest_model import run_random_forest
from models.LinearRegression_model import run_linear_regression



def main(modelname):
    """
    used this section to test models during project
    """
    if  modelname == 'LSTM':
        run_lstm()
    elif modelname=='KNN':
        run_knn()
    elif modelname=='RandomForest':
        run_random_forest()
    elif modelname=='Linear':
        run_linear_regression()
    elif modelname == 'Ensemble':
        run_ensemble()


if __name__=='__main__':
    model='KNN' # options are LSTM, KNN, RandForest, Linear
    main(model)

#