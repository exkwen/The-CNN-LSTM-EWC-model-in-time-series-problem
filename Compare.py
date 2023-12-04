from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


# Print relationship LSTM charts
def lstm_chart():
    lstm = pd.read_csv("experiment_results/LSTM.csv")
    lstm_1 = lstm.loc[lstm['filename'].str.contains("dataset/Stock_2.csv")]
    adam_lstm = lstm_1.loc[lstm['optimizer'].str.contains("Adam")]
    no_ewc_lstm = adam_lstm.loc[lstm['Use_EWC'].str.contains("No")]

    cnn_lstm_ewc = pd.read_csv("experiment_results/CNN-LSTM-EWC_1.csv")
    cnn_lstm_ewc.reset_index(drop=True, inplace=True)
    cnn_lstm_ewc['mean_absolute_error'].plot(label="CNN-LSTM-EWC")
    no_ewc_lstm.reset_index(drop=True, inplace=True)
    no_ewc_lstm['mean_absolute_error'].plot(label=" LSTM")

    plt.xlabel('Times')
    plt.ylabel('mean absolute error')
    plt.legend()
    plt.show()

    # plt.scatter(no_ewc_time_x, no_ewc_error_y, c="black", marker='s', label='centers')
    # plt.xlabel('times')
    # plt.ylabel('sepal width')
    # plt.legend(loc=2)
    plt.show()

    # lstm_3 = lstm_1.loc[lstm['optimizer'].str.contains("Adam")]
    # lstm_3 = lstm_3.loc[:, ['optimizer', 'times', 'train_loss', 'mean_absolute_error']]


if __name__ == '__main__':
    lstm_chart()
