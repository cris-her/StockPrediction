# Standard packages
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#sns.set_style('whitegrid')
#plt.style.use("fivethirtyeight")

# Scripts
from preprocess import config, get_timestamps, collect_data, plot_closing, plot_gain, compare_stocks
from models import TorchRNN, rnn_params
from dataset import GetDataset
from predictions import Classifier, plot_predictions

def visualization():
    for idx, stock in enumerate(config.stock_names):
        timestamps = get_timestamps(config.yrs, config.mths, config.dys)
        df = collect_data(timestamps, stock, config.moving_averages, True)
        fig1 = plot_closing(df, moving_averages=True, intervals=None)
        fig1.show()
        fig2 = plot_gain(df)
        fig2.show()
        daily_returns, fig1_c, fig2_c = compare_stocks(config.stock_names_compare, timestamps)


def make_predictions(features):
    timestamps = get_timestamps(config.yrs, config.mths, config.dys)
    if len(config.stock_names) == 1:
        for feature in features:
            df = collect_data(timestamps, config.stock_names[0], moving_averages=config.moving_averages, include_gain=True)
            dataset = GetDataset(df, feature=feature)
            dataset.get_dataset(scale=True)
            train_data, test_data, train_data_len = dataset.split(train_split_ratio=0.8, time_period=30)
            train_data, test_data = dataset.get_torchdata()
            x_train, y_train = train_data
            x_test, y_test = test_data
            params = rnn_params
            model = TorchRNN(rnn_type=params.rnn_type, input_dim=params.input_dim,
                            hidden_dim=params.hidden_dim, output_dim=params.output_dim,
                            num_layers=params.num_layers)
            clf = Classifier(model)
            clf.train([x_train, y_train], params=params, show_progress=True)
            scaler = dataset.scaler
            predictions = clf.predict([x_test, y_test], scaler, data_scaled=True)
            plot_predictions(df, train_data_len, predictions)




features = ['Adj Close']
#visualization()
make_predictions(features)













