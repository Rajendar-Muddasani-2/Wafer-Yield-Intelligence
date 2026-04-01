#!/usr/bin/env python
# coding: utf-8

"""
lot_stats.py - Third stage of ETL pipeline
Calculates statistical thresholds (mean, std, upper bounds) for yield metrics
"""

import pandas as pd
import cx_Oracle
import numpy as np
import os
from sqlalchemy import types, create_engine
from config_loader import Config
from database import DatabaseConnection


# --- Inputs ---



# Load configuration from .env file
config = Config()

lot_stats_cols = ["family", "product_type", "scope", "insertion", "sbin", "mean", "std", "default_UB_formula", "default_UB", "min_UB", "custom_UB"]
family = 'A1G'

# Initialize Oracle client from environment or default path
try:
    try:
        client_path = os.getenv('ORACLE_CLIENT_PATH', './oracle_client_lib')
        cx_Oracle.init_oracle_client(lib_dir=client_path)
        print('Initialised Oracle client library')
    except:
        client_path = os.getenv('ORACLE_CLIENT_PATH', '../oracle_client_lib')
        cx_Oracle.init_oracle_client(lib_dir=client_path)
        print('Initialised Oracle client library (parent dir)')
except Exception as e:
    print(e)
finally:
    pass


# --- 1. Read lot, lot_wafer, lot_wafer_site csv files ---



file_link = "lot_wafer_site.csv"

lot_wafer_site = pd.read_csv(file_link, sep=",", low_memory=False)
# reset index to start from 0
lot_wafer_site.reset_index(drop = True, inplace = True)


file_link = "lot_wafer.csv"

lot_wafer = pd.read_csv(file_link, sep=",", low_memory=False)
# reset index to start from 0
lot_wafer.reset_index(drop = True, inplace = True)


file_link = "lot.csv"

lot = pd.read_csv(file_link, sep=",", low_memory=False)
# reset index to start from 0
lot.reset_index(drop = True, inplace = True)


# --- 2. Read table from lot_info table ---



file_link = "lot_info.csv"

lot_info = pd.read_csv(file_link, sep=",", low_memory=False)
# reset index to start from 0
lot_info.reset_index(drop = True, inplace = True)

lot_info


# --- Clean df ---



def clean_df(df):
    # convert all cols to string
    df = df.map(str)
    # convert yield back to float
    df["yield"] = df["yield"].astype(float)
    
    return df


# --- Clean lot_data ---



lot_wafer_site = clean_df(lot_wafer_site)
# reorganise to desired format
lot_wafer_site = lot_wafer_site[["lot", "wafer", "site", "sbin", "measstep", "yield", "producttype"]]


lot_wafer = clean_df(lot_wafer)
# reorganise to desired format
lot_wafer = lot_wafer[["lot", "wafer", "sbin", "measstep", "yield", "producttype"]]


lot = clean_df(lot)
# rename scope to lot
#lot = lot.rename(columns={"scope": "lot"})
# reorganise to desired format
lot = lot[["lot", "sbin", "measstep", "yield", "producttype"]]


# --- Clean lot_info ---



# lot_info
lot_info = lot_info.map(str)
    # rename test_mode to measstep
lot_info = lot_info.rename(columns={"test_mode": "measstep"})
    # getting the actual measstep e.g. S1, S2
lot_info["measstep"] = lot_info["measstep"].astype(str).str[:2]
    # reorganise to desired format
lot_info = lot_info[["lot", "product_type", "measstep"]]
    # get the unqiue lot number for later filtering use
lot_num_unique = lot_info.lot.unique()
    # get the unqiue product type for later filtering use
lot_prod_type_unique = lot_info.product_type.unique()

lot_info


# --- 3. Filter df ---



def filter_df(df):
        #EBS15 lots
    #df = df[df["lot"].isin(["84PVFA00", "84P0X461"])]
    df = df[df["lot"].isin(lot_num_unique)]
        # M3990B00013
    #df = df[df["product_type"] == "M4283A00014"] 
    df = df[df["product_type"].isin(lot_prod_type_unique)]
        # SB0 > 60 %
    SB0 = df.loc[(df["sbin"] == "0") & (df["yield"] >= 0.6)]
        # NonSB0 < 20%
    nonSB0 = df.loc[(df["sbin"] != "0") | (df["yield"] <= 0.2)]
        #join filtered lot stats with sb0 and nonsb0
    df = pd.concat([SB0, nonSB0], ignore_index=True)
    
    return df


# --- Get mean and std ---



def get_mean_std(df, scope):
    criteria = ["measstep", "sbin", "product_type"]
    
    groupby_criteria = criteria.copy()
    
    if scope == "lot":
        new_criteria = "lot"
    elif scope == "wafer":
        new_criteria = "wafer"
    elif scope == "site":
        new_criteria = "site"
        
    groupby_criteria.append(new_criteria)
    
    # Step 1
        # get the mean of yield in lot level
    df = df.groupby(groupby_criteria)["yield"].agg(["mean"]).reset_index()

    #Step 2
        # get the mean and std of mean in lot level
    df = df.groupby(criteria)["mean"].agg(["mean", "std"]).reset_index()

    # replace NaN with 0
    df = df.fillna(0)
    
    return df


# --- Get default UB formula and calculate default UB ---



def get_default_UB(df, scope):
    sigma = 6
    
    if scope == "site":
        sigma = 7
    
    # insert default formula for upper bound
    df["default_UB_formula"] = f"mean + (mean * 0.01 + std * {sigma})"

    # calculate default upper_bound
    df["default_UB"] = df['mean'] + (df['mean'] * 0.01 + df['std'] * sigma)
    
    return df


# --- Get min UB ---



def get_min_UB(row):
    scope = row.scope
    default_UB = row.default_UB
    
    if scope == "lot":
        min_UB = 0.01
    elif scope == "wafer":
        min_UB = 0.01
    elif scope == "site":
        min_UB = 0.15
        
    if default_UB < min_UB:
        row["min_UB"] = min_UB
    else:
        row["min_UB"] = default_UB
        
    return row


# --- Get family ---



def get_family(df):
    df['family'] = family
    return df


# --- Convert to lot_stats format ---



def convert_format(df, scope):
    
    # join df to lot_info on lot and measstep
    df = df.set_index(['lot',"measstep"]).join(lot_info.set_index(['lot', "measstep", "product_type"]))
    # reset index
    df = df.reset_index()
    
    #filter df
    df = filter_df(df)
    
    # get mean and std
    df = get_mean_std(df, scope)

    # replace NaN with 0
    df = df.fillna(0)

    # insert scope
    df["scope"] = scope

    # get default UB formula and calculate UB
    df = get_default_UB(df, scope)

    # get min bound of UB
    df = df.apply(get_min_UB, axis = "columns")
    
    # get the family
    df = get_family(df)
    
    return df


# Convert lot_wafer_site



lot_wafer_site = convert_format(lot_wafer_site, "site")
lot_wafer_site


# Convert lot_wafer



lot_wafer = convert_format(lot_wafer, "wafer")
lot_wafer


# Convert lot



lot = convert_format(lot, "lot")
lot


# Join the dfs together



# join lot, lot_wafer, lot_wafer_site into one df
lot_stats = pd.concat([lot, lot_wafer, lot_wafer_site], ignore_index=True)

try:
    # insert user defined
    lot_stats["custom_UB"] = np.nan

    # rename measstep col into insertion
    lot_stats = lot_stats.rename(columns={"measstep": "insertion"})

    # round all numerical cols to 5 decimal points
    lot_stats = lot_stats.round(decimals = 5)

    # reorganise to desired format
    lot_stats = lot_stats[lot_stats_cols]

    # make all headers lowercase
    lot_stats.columns = map(str.lower, lot_stats.columns)

    # lot_stats.to_csv("lot_stats.csv", index = False)

except Exception as e:
    print(e)
    print("No records to insert.")

lot_stats


# --- 4. Insert df to Oracle database ---



db = DatabaseConnection(config)
engine = db.engine

# open db connection
connection = engine.connect()

# insert into lot_data
    # convert all object types to varchar to insert data faster (faster performance)
dtyp = {c:types.VARCHAR(lot_stats[c].str.len().max())
       for c in lot_stats.columns[lot_stats.dtypes == 'object'].tolist()}

    # append the existing data with this data into db
    # lot_stats.to_sql("lot_stats", engine, if_exists='append', dtype=dtyp, index=False)
lot_stats.to_sql("lot_stats", engine, if_exists='replace', dtype=dtyp, index=False)

    # make sure that lot_data has been inserted properly
print(pd.read_sql_table("lot_stats", connection))

connection.close()






