import pandas as pd
from pandas import ExcelWriter
import os
from datetime import datetime
from typing import List
import numpy as np

base_path = r"D:\Pucho\FlightDelay\Code\Data"
source_files_folder = os.path.join(base_path, "source")
train_file_names = ["Jan_2016.csv", "Feb_2016.csv", "Mar_2016.csv", 
    "Apr_2016.csv", "May_2016.csv", "Jun_2016.csv", "Jul_2016.csv", 
    "Aug_2016.csv", "Sep_2016.csv"]
test_file_names = ["Oct_2016.csv", "Nov_2016.csv"]
field_names = ["OP_CARRIER_AIRLINE_ID", "TAIL_NUM", "ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID",
    "CRS_DEP_TIME", "DEP_DELAY", "TAXI_OUT", "TAXI_IN", "CRS_ARR_TIME", "ARR_DELAY", 
    "AIR_TIME", "DISTANCE"]

airports = []
carriers = []
aircrafts = []

def generate_traintest_data(is_train_data: bool, train_file_names: List, field_names: List):
    global aircrafts
    global carriers
    global airports

    df_flightdelay_list = []
    file_generated = "train-data.csv"
    if not is_train_data:
        file_generated = "test-data.csv"
    for train_filename in train_file_names:
        filepath = os.path.join(source_files_folder, train_filename)
        df = pd.read_csv(filepath)
        df = df.loc[df['CANCELLED'] == 0]
        df = df.loc[df['FLIGHTS'] == 1]
        df = df[df['ARR_DELAY'].notnull()]
        df = df[df['ARR_DELAY'] <= 180]
        df = df[df['ARR_DELAY'] >= -180]
        column_list = df.columns.values.tolist()
        for column in column_list:
            if column not in field_names:
                del df[column]

        df_flightdelay_list.append(df)

    df_flightdelay = pd.concat(df_flightdelay_list)
    df_flightdelay.to_csv(os.path.join(base_path, file_generated), index=False)
    unique_carriers = df_flightdelay['OP_CARRIER_AIRLINE_ID'].unique()
    unique_tailnum = df_flightdelay['TAIL_NUM'].unique()
    unique_orig_airports = df_flightdelay['ORIGIN_AIRPORT_ID'].unique()
    unique_dest_airports = df_flightdelay['DEST_AIRPORT_ID'].unique()

    aircrafts.extend(unique_tailnum)
    aircrafts = list(set(aircrafts))
    carriers.extend(unique_carriers)
    carriers = list(set(carriers))
    airports.extend(unique_orig_airports)
    airports.extend(unique_dest_airports)
    airports = list(set(airports))

def generate_lookup_data():
    with open(os.path.join(base_path, 'aircrafts.txt'), 'w') as f:
        for item in aircrafts:
            f.write(f"{item}\n")

    with open(os.path.join(base_path, 'carriers.txt'), 'w') as f:
        for item in carriers:
            f.write(f"{item}\n")

    with open(os.path.join(base_path, 'airports.txt'), 'w') as f:
        for item in airports:
            f.write(f"{item}\n")


if __name__ == '__main__':
    generate_traintest_data(True, train_file_names, field_names)
    generate_traintest_data(False, test_file_names, field_names)
    generate_lookup_data()
    pass
    

