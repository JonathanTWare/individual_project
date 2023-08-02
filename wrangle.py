import pandas as pd
import numpy as np
import os
import zipfile
import subprocess
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from kaggle.api.kaggle_api_extended import KaggleApi


# -----------------------------acquire--------------------------------


def get_crime_data():
    destination_directory = '.'
    os.makedirs(destination_directory, exist_ok=True)

    csv_file_name = 'Crimes-2001-to-present-chicago.csv'
    csv_file_path = os.path.join(destination_directory, csv_file_name)

    if os.path.isfile(csv_file_path):
        
        df = pd.read_csv(csv_file_path)
    else:
        
        api = KaggleApi()
        api.authenticate()

        dataset_name = 'adelanseur/Crimes-2001-to-present-chicago'
        api.dataset_download_files(dataset_name, path=destination_directory, unzip=True)

        
        extracted_files = os.listdir(destination_directory)
        for file in extracted_files:
            if file.endswith('.csv'):
                local_file_path = os.path.join(destination_directory, file)
                break
        else:
            raise FileNotFoundError(f"CSV file '{csv_file_name}' not found in the extracted files.")

        os.replace(local_file_path, csv_file_path)

        
        df = pd.read_csv(csv_file_path)

    return df

 
# -----------------------------prep--------------------------------
def prep_crime_data(df):
    df = df.dropna()
    df = df[['Primary Type', 'Date']]
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)  
    five_years_ago = datetime.now() - timedelta(days=365 * 5)
    

    df = df[df.index >= five_years_ago]
    df = df.resample('D')['Primary Type'].value_counts()
    df = df.unstack(level='Primary Type', fill_value=0)
    df = df[['THEFT','BATTERY','ASSAULT', 'CRIMINAL DAMAGE','MOTOR VEHICLE THEFT','NARCOTICS','HOMICIDE','HUMAN TRAFFICKING','OFFENSE INVOLVING CHILDREN','KIDNAPPING']]
    
    
    
    

    
    return df
# -----------------------------split--------------------------------

    
def split_crime_data(df, train_percentage=0.6, validation_percentage=0.15):
    
    total_samples = len(df)
    train_size = int(total_samples * train_percentage)
    validation_size = int(total_samples * validation_percentage)

    train = df[:train_size]
    validation = df[train_size:train_size + validation_size]
    test = df[train_size + validation_size:]

    return train, validation, test


# -----------------------------wrangle-------------------------------- 

def wrangle_crime():

    df = get_crime_data()
    df = prep_crime_data(df)
    train, validate, test = split_crime_data(df, train_percentage=0.6, validation_percentage=0.15)
   
    return train, validate, test
