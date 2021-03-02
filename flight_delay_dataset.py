import torch
from torch.utils.data import Dataset, DataLoader
from common import base_data_path, cpu_device
import os
import pandas as pd
import numpy as np

#

class FlightDelayDataset(Dataset):

    def __init__(self, is_train_data: bool):
        super().__init__()
        if is_train_data:
            self.csv_path = os.path.join(base_data_path, "train-data.csv")
        else:
            self.csv_path = os.path.join(base_data_path, "test-data.csv")
        self.df_csv = None
        self.df_arr_delay = None

        self.len_df_csv = 0
        self.num_carrier = 0
        self.num_aircrafts = 0
        self.num_origin = 0
        self.num_dest = 0
        self.num_rest = 0

    def generate_lookup_data(self):
        with open(os.path.join(base_data_path, 'aircrafts.txt'), 'r') as f:
            aircrafts_list = f.read().splitlines()
            aircrafts_list.sort()
            index = 0
            aircrafts = {}
            for item in aircrafts_list:
                aircrafts[item] = index
                index += 1


        with open(os.path.join(base_data_path, 'carriers.txt'), 'r') as f:
            carriers_list = f.read().splitlines()
            carriers_list.sort()
            index = 0
            carriers = {}
            for item in carriers_list:
                carriers[item] = index
                index += 1

        with open(os.path.join(base_data_path, 'airports.txt'), 'r') as f:
            airports_list = f.read().splitlines()
            airports_list.sort()
            index = 0
            airports = {}
            for item in airports_list:
                airports[item] = index
                index += 1

        return airports, carriers, aircrafts

    def load_data(self):
        self.airports, self.carriers, self.aircrafts = self.generate_lookup_data()

        self.num_carrier = len(self.carriers)
        self.num_aircrafts = len(self.aircrafts)
        self.num_origin = len(self.airports)
        self.num_dest = len(self.airports)

        df_csv_data = pd.read_csv(self.csv_path)
        df_csv_data['DEP_TIME_SIN'] = np.sin(2 * np.pi * 
            ((df_csv_data['CRS_DEP_TIME'] // 100) * 60 + (df_csv_data['CRS_DEP_TIME'] -  df_csv_data['CRS_DEP_TIME'] // 100 * 100)) / 1440)
        df_csv_data['DEP_TIME_COS'] = np.cos(2 * np.pi * 
            ((df_csv_data['CRS_DEP_TIME'] // 100) * 60 + (df_csv_data['CRS_DEP_TIME'] -  df_csv_data['CRS_DEP_TIME'] // 100 * 100)) / 1440)
        df_csv_data['ARR_TIME_SIN'] = np.sin(2 * np.pi * 
            ((df_csv_data['CRS_ARR_TIME'] // 100) * 60 + (df_csv_data['CRS_ARR_TIME'] -  df_csv_data['CRS_ARR_TIME'] // 100 * 100)) / 1440)
        df_csv_data['ARR_TIME_COS'] = np.cos(2 * np.pi * 
            ((df_csv_data['CRS_ARR_TIME'] // 100) * 60 + (df_csv_data['CRS_ARR_TIME'] -  df_csv_data['CRS_ARR_TIME'] // 100 * 100)) / 1440)
        df_csv_data['DISTANCE'] = df_csv_data['DISTANCE'] / 10
        df_csv_data.drop(['CRS_DEP_TIME'], axis=1, inplace=True)
        df_csv_data.drop(['CRS_ARR_TIME'], axis=1, inplace=True)

        #rest_of_columns = ['DEP_TIME_SIN', 'DEP_TIME_COS', 'DEP_DELAY', 'TAXI_OUT', 'TAXI_IN', 'ARR_TIME_SIN', 'ARR_TIME_COS', 'DISTANCE']
        rest_of_columns = ['DEP_TIME_SIN', 'DEP_TIME_COS', 'ARR_TIME_SIN', 'ARR_TIME_COS', 'DISTANCE']
        self.num_rest = len(rest_of_columns)
        self.df_csv = df_csv_data
        self.df_arr_delay = df_csv_data['ARR_DELAY']
        self.len_df_csv = self.df_csv.shape[0]

    def __len__(self):
        return self.len_df_csv

    def __getitem__(self, index):
        if self.df_csv is not None:
            row = self.df_csv.iloc[index].to_dict()
            #input = [self.carriers[str(row['OP_CARRIER_AIRLINE_ID'])], self.aircrafts[row['TAIL_NUM']], 
            #    self.airports[str(row['ORIGIN_AIRPORT_ID'])], self.airports[str(row['DEST_AIRPORT_ID'])], 
            #    row['DEP_TIME_SIN'], row['DEP_TIME_COS'], row['DEP_DELAY'] / 60, row['TAXI_OUT'] / 60, row['TAXI_IN'] / 60, 
            #    row['ARR_TIME_SIN'],  row['ARR_TIME_COS'], row['DISTANCE'] / 600]

            input = [self.carriers[str(row['OP_CARRIER_AIRLINE_ID'])], self.aircrafts[row['TAIL_NUM']], 
                self.airports[str(row['ORIGIN_AIRPORT_ID'])], self.airports[str(row['DEST_AIRPORT_ID'])], 
                row['DEP_TIME_SIN'], row['DEP_TIME_COS'],  
                row['ARR_TIME_SIN'],  row['ARR_TIME_COS'], row['DISTANCE'] / 600]

            input = np.asarray(input, dtype=np.float32)
            output = row['ARR_DELAY'] / 60
            return torch.from_numpy(input).to(cpu_device), output 
        else:
            return None


            