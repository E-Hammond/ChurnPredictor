import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from PIL import Image
import pickle, os


def encoder(dataframe):

    for i in dataframe.index:
        if dataframe.status[i]=='Not Resolved':
            dataframe.status[[i]] = dataframe.status[[i]].replace('Not Resolved', 0)
        if dataframe.status[i]=='Resolved':
            dataframe.status[[i]] = dataframe.status[[i]].replace('Resolved', 1)
        if dataframe.age[i]=='18-29':
            dataframe.age[[i]] = dataframe.age[[i]].replace('18-29', 0)
        if dataframe.age[i]=='30-49':
            dataframe.age[[i]] = dataframe.age[[i]].replace('30-49', 1)
        if dataframe.age[i]=='50-64':
            dataframe.age[[i]] = dataframe.age[[i]].replace('50-64', 2)
        if dataframe.age[i]=='65+':
            dataframe.age[[i]] = dataframe.age[[i]].replace('65+', 3)
        if dataframe.complaint[i]=='No Coverage':
            dataframe.complaint[[i]] = dataframe.complaint[[i]].replace('No Coverage', 0)
        if dataframe.complaint[i]=='Slow Internet':
            dataframe.complaint[[i]] = dataframe.complaint[[i]].replace('Slow Internet', 1)
        if dataframe.complaint[i]=='USSD Errors':
            dataframe.complaint[[i]] = dataframe.complaint[[i]].replace('USSD Errors', 2)
        if dataframe.complaint[i]=='Poor Customer Service':
            dataframe.complaint[[i]] = dataframe.complaint[[i]].replace('Poor Customer Service', 3)
        if dataframe.complaint[i]=='High Call Charges':
            dataframe.complaint[[i]] = dataframe.complaint[[i]].replace('High Call Charges', 4)
        if dataframe.complaint[i]=='Network Interruptions':
            dataframe.complaint[[i]] = dataframe.complaint[[i]].replace('Network Interruptions', 5)
        if dataframe.complaint[i]=='Unsolicited Subscribe Messages':
            dataframe.complaint[[i]] = dataframe.complaint[[i]].replace('Unsolicited Subscribe Messages', 6)

    return dataframe


def standardise(dataframe):

    ## Minimize outliers by standardizing train set
    cat_var_df = dataframe.drop(columns=['monthly_charges','tenure'])
    num_var_df = pd.DataFrame(dataframe.loc[:,['monthly_charges','tenure',]])

    scalar = StandardScaler()
    num_var = scalar.fit_transform(num_var_df)

    ## Merge the dataframes after standardisation
    num_var_df = pd.DataFrame(num_var, columns=num_var_df.columns)
    dataframe = pd.concat(objs=[cat_var_df,num_var_df], axis=1)

    return dataframe



def encode_sample(dataframe):

    dataframe = dataframe.replace(['Yes','No','resolved','not resolved'],[1,0,1,0])
    dataframe['age'] = ordencoder.fit_transform(dataframe[['age']])
    dataframe['complaint'] = ordencoder.fit_transform(dataframe[['complaint']])
    
    return dataframe