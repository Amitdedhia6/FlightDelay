import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import openpyxl
import os
import pathlib
import spacy
from itertools import chain 
import argparse
import pandas as pd
import numpy as np
from common import device, google_cloud, base_model_path
from flight_delay_dataset import FlightDelayDataset
from network import Network


def train_model(batch_size, num_hidden_layers, activation_function, lr, weight_decay):
    train_set = FlightDelayDataset(True)
    train_set.load_data()
    
    nn_instance = Network(train_set.num_carrier, train_set.num_aircrafts, train_set.num_origin, train_set.num_dest, 
            train_set.num_rest, num_hidden_layers, activation_function).to(device)
    nn_instance.train()
    print(nn_instance)
    print(device)
    optimizer = optim.Adam(nn_instance.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

    for epoch in range(100):
        # get the next batch
        total_loss = 0

        for batch in train_loader:
            # get the next batch
            t_input, t_arr_delay = batch
            t_input = t_input.to(device, non_blocking=True)
            t_arr_delay = t_arr_delay.to(device, non_blocking=True)
            t_arr_delay = t_arr_delay.reshape(-1, 1)

            preds = nn_instance(t_input)
            loss = F.mse_loss(preds, t_arr_delay)
            total_loss += loss.item()

            # compute gradients using backprop
            optimizer.zero_grad() #so that the new grad do not accumulate over prev ones
            loss.backward()

            # update weights - using optimizer of choice
            optimizer.step()

           # print(nn_instance)
            pass

        if epoch % 1 == 0:
            print(f"epoch : {epoch}, current_loss: {total_loss}")
            

    #torch.save(nn_instance, os.path.join(base_model_path, network_param.get_model_filename))
    filename = str(batch_size) + "_" + str(num_hidden_layers) + "_" + str(activation_function) + "_" + str(lr) +  "_" + str(weight_decay) + ".pt"
    filepath = os.path.join(base_model_path, filename)
    torch.save(nn_instance, filepath)
    return filepath

def test_model(filepath):
    test_set = FlightDelayDataset(False)
    test_set.load_data()

    nn_instance = torch.load(filepath, map_location=device)
    nn_instance.eval()
    print(nn_instance)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)
    total_loss = 0
    total_good = 0
    final_pred = None
    for batch in test_loader:
        # get the next batch
        with torch.no_grad():
            t_input, t_arr_delay = batch
            t_input = t_input.to(device)
            t_arr_delay = t_arr_delay.to(device)
            t_arr_delay = t_arr_delay.reshape(-1, 1)

            preds = nn_instance(t_input)

            combine = torch.cat((t_arr_delay, preds), dim=1)
            if final_pred is None:
                final_pred = combine
            else:
                final_pred = torch.cat((final_pred, combine), dim = 0)

            loss = F.mse_loss(preds, t_arr_delay)
            total_loss += loss.item()
            t = preds - t_arr_delay
            total_good = total_good + np.sum(((t * 60 <= 10) & (t * 60 >= -10)).cpu().numpy())
            

    final_pred = final_pred * 60
    total_elements = final_pred.shape[0]
    np.savetxt(filepath + '.csv', np.asarray(final_pred.cpu()), delimiter=',', fmt='%1.1f')   
    accuracy =  total_good * 100 / total_elements
    print("     Accuracy %", accuracy)
    print("     Test Loss =", total_loss)

if __name__ == '__main__':
    # Main program starts here
    if 4 == 5:
        parser = argparse.ArgumentParser(description = 'FlightDelay Training Module')
        parser.add_argument("numLayers")
        parser.add_argument("batchSize")
        parser.add_argument("activationFunction")
        parser.add_argument("lr")
        parser.add_argument("weightDecay")

        args = parser.parse_args()
        num_layers = int(args.numLayers)
        batch_size = int(args.batchSize)
        activation_function = args.activationFunction
        lr = float(args.lr)
        weight_decay = float(args.weightDecay)

    if not google_cloud:
        num_layers = 1
        batch_size = 1000
        lr = 0.000001
        activation_function = 'sigmoid'
        weight_decay = 1

    #saved_model_path = train_model(batch_size, num_layers, activation_function, lr, weight_decay)
    #saved_model_path = os.path.join(base_model_path, "1_sigmoid_1e-06_1.pt")
    #test_model(saved_model_path)

    num_layers_list = [5, 4, 3, 2, 1]
    batch_size_list = [10000, 1000, 100]
    activation_function_list = ['relu', 'sigmoid', 'tanh']
    lr_list = [0.000001, 0.00001, 0.0001, 0.001]
    weight_decay = 0

    num_layers_list = [3]
    batch_size_list = [1000]
    activation_function_list = ['relu']
    lr_list = [0.000001]
    weight_decay = 0

    for num_layer in num_layers_list:
        for batch_size in batch_size_list:
            for activation_function in activation_function_list:
                for lr in lr_list:
                    saved_model_path = train_model(batch_size, num_layer, activation_function, lr, weight_decay)
                    #saved_model_path = os.path.join(base_model_path, "1_sigmoid_1e-06_1.pt")
                    test_model(saved_model_path)
    