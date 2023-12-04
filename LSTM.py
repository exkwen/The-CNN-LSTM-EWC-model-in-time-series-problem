from __future__ import print_function
import argparse
from copy import deepcopy

import numpy as np
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from pandas import read_csv
from torch.autograd import Variable, variable
import os
import torch.nn.functional as F


# data preprocessing
def clean_data(args, filename, learned_percent):
    stock_data = read_csv(filename)
    # obtain the necessary features
    name_col = ["close", "open", "high", "low", "volume"]
    stock_data = stock_data[name_col]
    close_max = stock_data['close'].max()
    close_min = stock_data['close'].min()

    # normalize the values in feature [0, 1]
    df = stock_data.apply(lambda x: (x - min(x)) / (max(x) - min(x)))
    sequence = args.seq_len
    X = []
    Y = []
    # previous sequence day is input feature, sequence+1 is table df.shape[0]
    for i in range(df.shape[0] - sequence - 1):
        X.append(np.array(df.iloc[i:(i + sequence), :].values, dtype=np.float32))
        Y.append(np.array(df.iloc[i + sequence + 1, 0], dtype=np.float32))

    total_len = len(Y)

    # divided the dataset into training set and testing set
    train_x, train_y = X[:int(learned_percent * total_len)], Y[:int(learned_percent * total_len)]
    test_x, test_y = X[int(0.99 * total_len):], Y[int(0.99 * total_len):]

    # design a suitable dataset, model can accept it
    train_loader = DataLoader(dataset=Mydataset(train_x, train_y, transform=transforms.ToTensor()),
                              batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=Mydataset(test_x, test_y), batch_size=args.batch_size, shuffle=True)
    return close_max, close_min, train_loader, test_loader


# set a suitable dataset from LSTM to read
class Mydataset(Dataset):
    def __init__(self, xx, yy, transform=None):
        self.x = xx
        self.y = yy
        self.tranform = transform

    def __getitem__(self, index):
        x1 = self.x[index]
        y1 = self.y[index]
        if self.tranform is not None:
            return self.tranform(x1), y1
        return x1, y1

    def __len__(self):
        return len(self.x)


class LSTM(nn.Module):

    def __init__(self, args):
        super(LSTM, self).__init__()
        # lstm的输入 #batch,seq_len, input_size
        self.hidden_size = args.hidden_size
        self.input_size = args.input_size
        self.num_layers = args.num_layers
        self.output_size = args.output_size
        self.seq_len = args.seq_len
        self.dropout = args.dropout
        self.batch_first = args.batch_first
        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                           batch_first=self.batch_first, dropout=self.dropout)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        if self.seq_len == 1:
            x = x.view(len(x), 1, -1)
        out, (hidden, cell) = self.rnn(x)
        out = self.linear(hidden)
        return out


# EWC algorithm
class EWC(object):
    def __init__(self, model, dataloader, device, importance=50):
        self.model = model
        self.importance = importance
        self.device = device
        self.params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher(dataloader)

    # 计算fisher信息矩阵
    def _compute_fisher(self, dataloader):
        fisher = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                fisher[n] = torch.zeros_like(p.data)

        loss_fun = nn.MSELoss(reduction='sum')

        self.model.eval()
        for idx, (data, label) in enumerate(dataloader):
            self.model.zero_grad()
            data, target = data.to(self.device), label.to(self.device)
            pred = self.model(Variable(data))
            pred = pred.squeeze(-1)
            pred = pred.permute(1, 0)
            label = label.unsqueeze(1)
            loss = loss_fun(pred, label)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    fisher[n] += (p.grad ** 2) / len(dataloader)

        return fisher

    def penalty(self, new_model):
        loss = 0
        for n, p in new_model.named_parameters():
            if p.requires_grad:
                _loss = self.fisher[n] * (p - self.params[n]) ** 2
                loss += _loss.sum()
        return loss * (self.importance / 2)


# build the default commends
def arg_default():
    parser = argparse.ArgumentParser(description='PyTorch LSTM Example')

    # data preprocessing
    parser.add_argument('--seq_len', type=int, default=1, help='the collection information of a few days')
    parser.add_argument('--useGPU', default=False, type=bool, help='whether useGPU to run the code')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed for shuffling the dataset')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

    # LSTM settings
    parser.add_argument('--num_layers', default=1, type=int, help='LSTM layers')
    parser.add_argument('--input_size', default=5, type=int, help='Enter the dimension of the feature')
    parser.add_argument('--hidden_size', default=32, type=int, help='Dimension of hidden layer')
    parser.add_argument('--output_size', type=int, default=1, help='the number of predicted value')
    parser.add_argument('--batch_first', default=True, type=bool, help='batch_first (default: True)')
    parser.add_argument('--dropout', default=0, type=float,  help='dropout (default: 0.1)')

    # model training settings
    parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.02, metavar='LR', help='learning rate')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='Learning rate step gamma')

    # the place of saving models
    parser.add_argument('--save_file', default='model/LSTM_Stock.pkl', help='used to save the current model')

    args = parser.parse_args()

    return args


# model training
def train(args, model, train_loader, optimizer, device, epoch, ewc=None, ewc_lambda=0.0):
    loss_fun = nn.MSELoss(reduction='sum')

    train_loss = 0

    for idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        # use GPU to run the code ( can accelerate)
        if args.useGPU:
            pred = model(Variable(data).cuda())
            pred = pred.squeeze(-1)
            pred = pred.permute(1, 0)
            label = label.unsqueeze(1).cuda()

        # use CPU
        else:
            pred = model(Variable(data))
            pred = pred.squeeze(-1)
            pred = pred.permute(1, 0)
            label = label.unsqueeze(1)

        loss = loss_fun(pred, label)

        # judge whether use the EWC algorithm
        if ewc is not None:
            ewc_loss = ewc.penalty(model)
            loss += ewc_lambda * ewc_loss

        train_loss += loss.item()

        # update the weights
        loss.backward()
        optimizer.step()

        # the beginning of each batch will show the information
        print('Train Epoch: {} [{}/{} ({:.0f}%) batch_size: {}]\tLoss: {:.6f}'.format(
            epoch, idx, len(train_loader), 100. * idx / len(train_loader), len(data), loss.item()))

    average_train_loss = train_loss / len(train_loader)
    print("The final avarage train loss: {:.6f}".format(average_train_loss))
    average_train_loss = ('%.6f' % average_train_loss)
    # save the trained model
    torch.save({'state_dict': model.state_dict()}, args.save_file)

    return average_train_loss


# model testing
def test(args, model, close_max, close_min, test_loader, device):
    checkpoint = torch.load(args.save_file)
    model.load_state_dict(checkpoint['state_dict'])
    preds = []
    labels = []
    with torch.no_grad():
        for idx, (x, label) in enumerate(test_loader):
            x, label = x.to(device), label.to(device)
            if args.useGPU:
                x = x.squeeze(1).cuda()  # batch_size,seq_len,input_size
            else:
                x = x.squeeze(1)
            pred = model(x)
            list = pred.data.squeeze(1).tolist()
            preds.extend(list[-1])
            labels.extend(label.tolist())

    absolute_error = 0
    # show the predicted results and true labels
    for i in range(len(preds)):
        print('prediction value: %.2f,  label value: %.2f' % (preds[i][0] * (close_max - close_min) + close_min,
                                                              labels[i] * (close_max - close_min) + close_min))
        absolute_error += abs(
            (preds[i][0] * (close_max - close_min) + close_min) - (labels[i] * (close_max - close_min) + close_min))

    mean_absolute_error = absolute_error/len(preds)
    print('Test set: mean absolute_error: {:.4f}\n'.format(mean_absolute_error))
    mean_absolute_error = ('%.5f' % mean_absolute_error)
    return mean_absolute_error


# record experimental results in a csv file
def collect_inform(inform):
    filename = 'experiment_results/LSTM.csv'
    field_order = ["filename", 'Use_EWC', "optimizer", 'scheduler', 'learned_percent', 'learning_rate', 'times',
                   'average_train_loss', 'mean_absolute_error']

    # create csv file
    flag = os.path.exists(filename)
    if not flag:
        with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
            csv_write = csv.DictWriter(f, field_order)
            csv_write.writeheader()

    # write experiment results
    with open(filename, 'a+', newline='', encoding='utf-8-sig') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(inform)

    # delete the duplicates
    data = pd.read_csv(filename)
    data.drop_duplicates(inplace=True)
    data.to_csv(filename, index=False)


def lstm(filename, op_name, time_step, le_rate, learned_percent):
    # set the hyper-parameters
    args = arg_default()
    args.lr = le_rate
    args.seq_len = time_step

    # shuffle dataset and preprocessing
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.useGPU else "cpu")
    close_max, close_min, train_loader, test_loader = clean_data(args, filename, learned_percent)

    model = LSTM(args)

    # choose the optimizer to find the min loss values, set the learning rate
    # StepLR can control the reduce of learning rate (Equal interval adjustment)
    if op_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # train the CNN model in epoch times and record the experimental results
    print("not use EWC")
    for epoch in range(1, args.epochs + 1):
        average_train_loss = train(args, model, train_loader, optimizer, device, epoch)
        mean_absolute_error = test(args, model, close_max, close_min, test_loader, device)
        scheduler.step()

        # record experimental results
        inform = [filename, 'No', op_name, "StepLR:1", learned_percent, args.lr, epoch, average_train_loss,
                  mean_absolute_error]
        collect_inform(inform)

    # print(" Use EWC")
    # for epoch in range(1, args.epochs + 1):
    #     ewc = EWC(model, train_loader, device)
    #     average_train_loss = train(args, model, train_loader, optimizer, device, epoch, ewc=ewc, ewc_lambda=0.01)
    #     mean_absolute_error = test(args, model, close_max, close_min, test_loader, device)
    #     scheduler.step()
    #     # experimental results
    #     inform = [filename, 'Yes', op_name, "StepLR:1", learned_percent, args.lr, epoch, average_train_loss,
    #               mean_absolute_error]
    #     collect_inform(inform)


if __name__ == '__main__':
    # filename, optimizer name, learning rate, learned percent
    lstm("dataset/Stock_1.csv", "Adam", 1, 0.001, 0.99)





