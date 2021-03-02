import os
import torch
import pathlib

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
cpu_device = torch.device("cpu") 
google_cloud = True

base_data_path =  os.path.join(pathlib.Path().absolute(), 'Data')
base_model_path =  os.path.join(pathlib.Path().absolute(), 'Models')

if google_cloud:
    base_data_path = os.path.join(r'/home/amitudedhia/FlightDelay/Code', 'Data')
    base_model_path = os.path.join(r'/home/amitudedhia/FlightDelay/Code', 'Models')