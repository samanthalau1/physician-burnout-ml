# functions for data loading and processing

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# reads data from an excel sheet
def load_data(file_path):
    data = pd.read_excel(file_path)
    return data


# remove unecessary data and rename
def clean(data):
    clean_data = data.drop(columns=['Respondent ID', 'Collector ID', 'Start Date', 'End Date', 'IP Address', 'Email Address', 
                                    'First Name', 'Last Name', 'Custom Data 1'])
    clean_data = clean_data.drop(index=0, axis=0)
    clean_data.columns=['Age', 'Gender', 'Specialty', 'Practice Type', 'Practice Size', 'New Patients', 'Years Worked',
                        'Patient Hours', 'EHR Hours', 'Admin Hours', 'Income Change', 'Burnout Level']

    clean_data.dropna(subset=['Burnout Level'], inplace=True, how='any')
    return clean_data


# cleaning words from burnout level
def extractNum(text):
    if isinstance(text, int):
        return text
    
    num = re.findall(r'\d+', str(text))
    return int(num[0])


# changes strings to numeric values
def fix_times(time, ogTime, fixTo):
    for i in range(len(ogTime)):
        if(ogTime[i] == time):
            return fixTo[i]
    return time


# fixes hours columns for graphing
def fix_hours(clean_data):

    # fixing hours columns for graphing
    patHours = ["Less than 20", "20-29", "30-39", "40-49", "50+"]
    EHR_AdmHours = ["Less than 5", "5-10", "11-15", "16+"]
    fixedPat = [10, 24.5, 34.5, 44.5, 50]
    fixedEHRAdm = [2, 7.5, 13, 16]

    graph_data = clean_data.copy()

    graph_data['Patient Hours'] = graph_data['Patient Hours'].apply(lambda hours: fix_times(hours, patHours, fixedPat))
    graph_data['EHR Hours'] = graph_data['EHR Hours'].apply(lambda hours: fix_times(hours, EHR_AdmHours, fixedEHRAdm))
    graph_data['Admin Hours'] = graph_data['Admin Hours'].apply(lambda hours: fix_times(hours, EHR_AdmHours, fixedEHRAdm))
    return graph_data


# fixes years for graphing 
def fix_years(clean_data):

    fixedYears = [0, 3, 8, 13, 16]
    years = ["Last 12 months", "1-5 years", "6-10 years", "11-15 years", "16+ years"]

    graph_data = clean_data.copy()
    graph_data['Years Worked'] = graph_data['Years Worked'].apply(lambda year: fix_times(year, years, fixedYears))
    return graph_data


# collapsed burnout level to low, medium, high
def collapse(level):
    if(level == 1 or level == 2):
        return "Low"
    if(level == 3):
        return "Moderate"
    if(level == 4 or level == 5):
        return "High"
    return level


# split train and test data 80/20
def split(clean_data):
    y = clean_data[['Burnout Level']].copy()
    X = clean_data.drop(columns=['Burnout Level'])

    y['Burnout Level'] = y['Burnout Level'].apply(lambda level: collapse(level))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    return X_train, X_test, y_train, y_test


# one hot encoding X
def encode(X_train, X_test):
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_train_encoded = encoder.fit_transform(X_train)
    X_test_encoded = encoder.transform(X_test)
    return X_train_encoded, X_test_encoded
