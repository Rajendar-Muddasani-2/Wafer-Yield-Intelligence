#!/usr/bin/env python
# coding: utf-8

"""
lot_info.py - First stage of ETL pipeline
Extracts lot information from raw test data and stores in database
"""

import pandas as pd
import cx_Oracle
import os
from sqlalchemy import types, create_engine
from config_loader import Config
from database import DatabaseConnection


# --- 1. Inputs ---



# Load configuration from .env file
config = Config()

# define df
existing_lot_info = None

# select csv file to read
file_name = "<DATA_PATH_FROM_ENV>"  # Set via INPUT_DATA_PATH env var

# select desired cols from raw data
cols = ["lot", "wafer", "site_no", "sbin", "hbin", "producttype", "measstep", "submeasstep", "meascategory", "testprgname", "begintimestamp;datetime", "endtimestamp;datetime"]

lot_info_cols = ["task_id", "product_type", "lot", "package", "test_mode", "test_program", "begin_timestamp", "end_timestamp"]

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


# --- 2. Read table from lot_info to get taskid and determine duplicate records ---



try:
    print('connecting to oracle database')
    db = DatabaseConnection(config)
    engine = db.engine
    
    # open db connection
    connection = engine.connect()

    # read lot_info table --name: LOTT_INFO_A1G_TEST
    existing_lot_info = pd.read_sql_table("lot_info", connection)
    print(existing_lot_info)
    
    # get task id by max id of lot_data
    task_id = existing_lot_info.task_id.max() + 1
    
    if pd.isnull(task_id):
        task_id = 1
    
    print('done taking existing lot info')
    
except Exception as e:
    print(e)
    
    task_id = 1
    
finally:
    connection.close()


# --- Read and clean data ---

# Clean header



def get_clean_header(header_list):
    header_dict = {}
    
    [header_dict.update({each : each[:each.index(";")]}) for each in header_list]
    
    return header_dict


# Read data method



def read_data(file_link):
    df = pd.read_csv(file_link + ".txt", sep=",", low_memory=False, dtype=str)
    
    # Find the columns where each value is null
    empty_cols = [col for col in df.columns if df[col].isnull().all()]
    # Drop all empty columns from the dataframe
    df = df.drop(empty_cols, axis=1)

    # convert all cols in df into string
    df = df.map(str)

    # make header lowercase
    df.columns = map(str.lower, df.columns)

    return df


# 3. Reading the data and arranging the header



# read the data
print('reading input data...')
original_df = read_data(file_name)

# get the first 11 characters only
original_df["producttype"] = original_df["producttype"].astype(str).str[:11]

# the code below, to move the 5 col from in the middle to the end. Not sure why the EBS data slot the 5 col in the middle,...
# so i move it to the end
original_df = original_df[[c for c in original_df if c not in ['lbin_p', 'lbin_p_name', 'hbin', 'sbin', 'site_no']] 
       + ['lbin_p', 'lbin_p_name', 'hbin', 'sbin', 'site_no']]

# get tasdef as list to clean it
old_header = original_df.iloc[:,28:-5].columns.to_list() #edited here
#old_header
original_df

print('done reading input data')


# Rename header accordingly



# clean tasdef header
header_dict = get_clean_header(old_header)
df = original_df.rename(columns = header_dict)
print('renaming column header done')
df


# Check for duplicate records



def get_new_records(row, df):    
    criteria = {"producttype":[row.product_type], "lot":[row.lot], "test_mode":[row.test_mode], "testprgname":[row.test_program]}
    
    # retrieve records from lot_data that has the same scope, insertion and sbin as the row
    mask = df[df[["producttype", "lot", "test_mode", "testprgname"]].isin(criteria).all(axis=1)]
    
    if len(mask.index) != 0:
        
        begin_time = row.begin_timestamp
        end_time = row.end_timestamp
    
        # check if timestamp is between begin and end timestamp of row timestamp
        mask = mask[mask["begintimestamp;datetime"] >= begin_time]
        mask = mask[mask["endtimestamp;datetime"] <= end_time]
        
        # if records exist within begin and end timestamp, change exist col to True
        if (len(mask.index) != 0):
            df.iloc[list(mask.index),-1] = True




if not existing_lot_info.empty:
    print('checking new record')
    lot_info = existing_lot_info.copy()
    
    # convert timestamp to datetime format
    lot_info["begin_timestamp"] = pd.to_datetime(lot_info["begin_timestamp"])
    lot_info["end_timestamp"] = pd.to_datetime(lot_info["end_timestamp"])
     
    # get corresponding letter for meascategory
    df.meascategory = df.meascategory.replace({"PRODUCTIVE":"P", "ENGINEERING":"E", "ANALYSIS":"X" })
    # concatenate measstep, submeasstep and meascategory and put under test_mode column
    df["test_mode"] = df[["measstep", "submeasstep", "meascategory"]].agg(''.join, axis=1)

    # convert timestamp to datetime format
    df["begintimestamp;datetime"] = pd.to_datetime(df["begintimestamp;datetime"])
    df["endtimestamp;datetime"] = pd.to_datetime(df["endtimestamp;datetime"])
    
    # add exist col to determine new records
    df["exist"] = False
    
    # get new records only
    lot_info.apply(get_new_records, axis = "columns", df = df)
    
    # filter all new records
    df = df[df['exist'] == False]
    
    # drop "test_mode" and "exist" col
    df = df.drop(["test_mode", "exist"], axis = 1)
    
    # convert timestamps back to string
    df = df.map(str)
    
    print('checking duplicate done')
    df
else:
    print('DataFrame is empty')


# 4. Saving the df into csv



# if df.empty != True:
#     df.to_csv("new_records_only.csv", index = False)
#     print('save df to csv file done')

df.to_csv("new_records_only.csv", index = False)
print('save df to csv file done')


# Select columns



if not df.empty:
    # choose and reorder header
    df = df[cols]

    print(df)


# --- Get relevant records ---



def get_mask(row, *args):
    args = list(args)

    criteria = {"lot":[row.lot], "measstep":[row.measstep]}
    
    # criteria for lot_info
    if "testprgname" in args:
        criteria["producttype"] = [row.producttype]
        criteria["test_mode"] = [row.test_mode]
        criteria["testprgname"] = [row.testprgname]
    else:
        # lot
        criteria["sbin"] = [row.sbin]

        # criteria for lot_data
        if "wafer" in args:
            # lot_wafer
            criteria["wafer"] = [row.wafer]

            # lot_wafer_site
            if "site_no" in args:
                criteria["site_no"] = [row.site_no]
                
    
    # filter the rows with the criteria
    mask = df[df[args].isin(criteria).all(axis=1)].reset_index()

    return mask


# --- Get timestamps ---



def calculate_timestamps(row, *args):
    mask = get_mask(row, *args)
    
    # get the timestamps
    row["begin_timestamp"] = mask.loc[:, "begintimestamp;datetime"].min()
    row["end_timestamp"] = mask.loc[:, "endtimestamp;datetime"].max()

    return row


# --- 5. Convert to lot_info format ---



if df.shape[0] != 0:
    print('converting to lot_info format')
    lot_info = df.copy()

    # define groupby criteria
    criteria = ["producttype", "lot", "test_mode", "testprgname"]

    # get corresponding letter for meascategory
    lot_info.meascategory = lot_info.meascategory.replace({"PRODUCTIVE":"P", "ENGINEERING":"E", "ANALYSIS":"X" })

    # concatenate measstep, submeasstep and meascategory and put under test_mode column
    lot_info["test_mode"] = lot_info[["measstep", "submeasstep", "meascategory"]].agg(''.join, axis=1)

    # need to add to df so later can find teststamps
    df["test_mode"] = lot_info["test_mode"]

    # get unique sets
    lot_info = lot_info.drop_duplicates(subset = criteria).reset_index()

    # insert taskid
    lot_info["task_id"] = task_id
    # insert package
    lot_info["package"] = "wafer"

    # calculate timestamps
    lot_info = lot_info.apply(calculate_timestamps, args=(criteria), axis = "columns")

    # rename columns
    lot_info = lot_info.rename(columns={"producttype": "product_type", "testprgname": "test_program"})

    # reorganise to desired format
    lot_info = lot_info[lot_info_cols]
    
    lot_info.to_csv("lot_info.csv")
    lot_info
    
    print('done converting to lot_info format')


# --- 6. Insert df to Oracle database ---



if df.empty != True:

    engine = create_engine(f'oracle+cx_oracle://{username}:{password}@{tns}', max_identifier_length=128)
    # engine = create_engine(f'oracle+cx_oracle://{username}:{password}@{tns}', max_identifier_length=128, echo='debug')

    # open db connection
    connection = engine.connect()

    # insert into lot_info
        # convert all object types to varchar to insert data faster (faster performance)
    dtyp = {c:types.VARCHAR(lot_info[c].str.len().max())
           for c in lot_info.columns[lot_info.dtypes == 'object'].tolist()}

        # append the existing data with this data into db
    lot_info.to_sql("lot_info", engine, if_exists="append", dtype=dtyp, index=False)
    # lot_info.to_sql("lot_info", engine, if_exists='replace', dtype=dtyp, index=False)

        # make sure that lot_data has been inserted properly
    print(pd.read_sql_table("lot_info", connection))
    
    connection.close()




print('entire lot info script done')






