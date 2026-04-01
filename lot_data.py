#!/usr/bin/env python
# coding: utf-8

"""
lot_data.py - Second stage of ETL pipeline
Calculates yield metrics for lots, wafers, and sites
"""

import pandas as pd
import numpy as np
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
file_link = "new_records_only.csv"

# select desired cols from raw data
cols = ["lot", "wafer", "site_no", "sbin", "hbin", "producttype", "measstep", "submeasstep", "meascategory", "testprgname", "begintimestamp;datetime", "endtimestamp;datetime"]

lot_data_cols = ["task_id", "lot", "wafer", "site", "insertion", "hbin", "sbin", "yield", "offset", "qty"]

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


# --- 2. Read table from lot_info to get taskid, and lot_stats table to calculate offset ---



try:
    db = DatabaseConnection(config)
    engine = db.engine
    
    # open db connection
    connection = engine.connect()
    
    # read lot_info table
    existing_lot_info = pd.read_sql_table("lot_info", connection)
    print("Retrieved existing lot_info successfully.")
    
    # get task id by max id + 1
    task_id = existing_lot_info.task_id.max()
    
    # read lot_stats table
    lot_stats = pd.read_sql_table("lot_stats", connection)
    print("Retrieved lot_stats successfully.")
    
except Exception as e:
    print(e)
    
finally:
    connection.close()


# --- Read data and select columns ---



df = pd.read_csv(file_link, sep=",", low_memory=False)

if df.empty:
    print('df is empty, quitting script')
    quit()

print('reading new records csv file')

# choose and reorder header
df = df[cols]

df.reset_index(drop = True, inplace = True)

# convert all cols in df into string
df = df.map(str)

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


# --- Calculate yield ---



def calculate_yield(row, *args):
    # e.g. lot level: COUNTIFs(lot, sbin)
    wSBIN = get_mask(row, *args)
    
    # e.g. lot level: COUNTif(lot) 
    arg = list(args)
    arg.remove("sbin")
    arg = tuple(arg)
    
    woSBIN = get_mask(row, *arg)
    
    # e.g. lot level: COUNTIFs(lot, sbin)/COUNTif(lot) 
    row["yield"] = len(wSBIN.index) / len(woSBIN.index)

    return row


# --- Calculate offset ---



def calculate_offset(row):  
    
    # determine scope using # of underscores (_)
    scope = "lot"
    if row["scope"].count("_") == 1:
        scope = "wafer"
    elif row["scope"].count("_") == 2:
        scope = "site"

    # filter scope
    mask = lot_stats.loc[lot_stats["scope"] == scope]
        
    # filter insertion and sbin
    mask = mask.loc[(mask["insertion"] == row.measstep) & (mask["sbin"] == row.sbin)].reset_index(drop = True)
    
    if not mask.empty:
        # priority: user define -> min UB
        if pd.isna(mask.at[0,"custom_ub"]):
            UB = mask.at[0,"min_ub"]
        else:
            UB = mask.at[0,"custom_ub"]            
            
    else:
        if scope == "lot":
            UB = 0.01
        elif scope == "wafer":
            UB = 0.01
        elif scope == "site":
            UB = 0.15
            
    row["offset"] = UB - row["yield"]
        
    return row


# --- 3. Convert to lot_data format ---

# Convert lot_wafer_site



if df.empty != True:
    print('converting to lot_wafer_site')
    # define groupby criteria
    criteria = ["lot", "wafer", "site_no", "sbin", "measstep", "hbin", "producttype"]
    # count each group (lot, wafer, site, measstep) and put under the column count
    lot_wafer_site = df.groupby(criteria).size().reset_index().rename(columns={0:'count'})
    # use a tempdf
    tempdf = lot_wafer_site.copy()
    # drop the col hbin if not will cause error as it is input for the yield calculation
    tempdf.drop(columns=['hbin', 'producttype'])
    
    # calculate yield
    lot_wafer_site = tempdf.apply(calculate_yield, args=(["lot", "wafer", "site_no", "sbin", "measstep"]), axis = "columns")
    # concatenate lot, wafer, site and put under Scope column
    lot_wafer_site['scope'] = lot_wafer_site[["lot", "wafer", "site_no"]].agg('_'.join, axis=1)
    
    # rename the col count to qty
    lot_wafer_site = lot_wafer_site.rename(columns={"count": "qty", "site_no": "site"})
    # make the temp df empty, should be able to save some space
    tempdf=None
    
    lot_wafer_site.to_csv("lot_wafer_site.csv")
    lot_wafer_site
    
    print('done converting to lot_wafer_site and saved as csv')
else:
    print('df empty, nothing to convert to lot_wafer_site')


# Convert lot_wafer



if df.empty != True:
    print('converting to lot_wafer')
    # define groupby criteria
    criteria = ["lot", "wafer", "sbin", "measstep", "hbin", "producttype"]
    # count each group (lot, wafer) and put under the column count
    lot_wafer = df.groupby(criteria).size().reset_index().rename(columns={0:'count'})
    # use a tempdf
    tempdf = lot_wafer.copy()
    # drop the col hbin if not will cause error as it is input for the yield calculation
    tempdf.drop(columns=['hbin', 'producttype'])
    
    # calculate yield
    lot_wafer = tempdf.apply(calculate_yield, args=(["lot", "wafer", "sbin", "measstep"]), axis = "columns")
    # concatenate lot, wafer and put under Scope column
    lot_wafer['scope'] = lot_wafer[["lot", "wafer"]].agg('_'.join, axis=1)
    
    # rename the col count to qty
    lot_wafer = lot_wafer.rename(columns={"count": "qty"})
    # empty the tempdf
    tempdf=None

    lot_wafer.to_csv("lot_wafer.csv")
    lot_wafer
    
    print('done converting to lot_wafer and saved as csv')
else:
    print('df empty, nothing to convert to lot_wafer')


# Convert lot



if df.empty != True:
    print('converting to lot')
    # define groupby criteria
    criteria = ["lot", "sbin", "measstep", "hbin", "producttype"]
    # count each lot and put under the column count
    lot = df.groupby(criteria).size().reset_index().rename(columns={0:'count'})
    # use a tempdf
    tempdf = lot.copy()
    # drop the col hbin if not will cause error as it is input for the yield calculation
    tempdf.drop(columns=['hbin', 'producttype'])
    
    # calculate yield
    lot = tempdf.apply(calculate_yield, args=(["lot", "sbin", "measstep"]), axis = "columns")
    # rename Lot col into Scope
    #lot = lot.rename(columns={"lot": "scope"})
    lot['scope'] = lot.loc[:, 'lot']
    
    # rename the col count to qty
    lot = lot.rename(columns={"count": "qty"})
    # empty the tempdf
    tempdf=None

    lot.to_csv("lot.csv")
    lot
    
    print('done converting to lot and saved as csv')
else:
    print('df empty, nothing to convert to lot')


# Join the dfs together



if df.empty != True:
    print('joining dfs together')
    # join lot, lot_wafer, lot_wafer_site into one df
    lot_data = pd.concat([lot, lot_wafer, lot_wafer_site], ignore_index=True)

    try:
        # change nan to none as database might not recognise nan
        lot_data=lot_data.where(lot_data.notna(), '')
        
        # convert all cols in lot_wafer_site into string
        lot_data = lot_data.map(str)
        # convert yield back to float in lot_wafer_site
        lot_data["yield"] = lot_data["yield"].astype(float)

        # calculate offset
        lot_data = lot_data.apply(calculate_offset, axis = "columns")

        # insert taskid
        lot_data["task_id"] = task_id

        # rename measstep col into insertion
        lot_data = lot_data.rename(columns={"measstep": "insertion"})

        # round all numerical cols to 5 decimal points
        lot_data = lot_data.round(decimals = 5)
        
        # drop the scope column for insertion into db
        lot_data.drop(columns=['scope'])

        # reorganise to desired format
        lot_data = lot_data[lot_data_cols]
        
        print('done joining all into one df')

    except Exception as e:
        #print("No records to be converted.")
        print(e)

    # lot_data.to_csv("lot_data.csv")
    lot_data


# Originally plan to compute the remaining SB into 'others' fill but decided to do it on the fly with the webapp



# # the function to get the sum value of the remaining as others
# hbin_others, sbin_others = 'others', 'others'

# def others_row_val(input_df):
#     input_df = input_df.astype({'sbin':int, 'qty':int})
#     input_df = input_df[input_df.sbin>0].reset_index(drop=True)
#     input_df.sort_values(by='qty', ascending=False).reset_index(drop=True)
#     #print(input_df)
#     input_df = input_df.loc[30:].reset_index(drop=True)
#     get_insertion = input_df.insertion.unique()
#     sum_yield = input_df['yield'].sum()
#     sum_offset = input_df['offset'].sum()
#     sum_qty = input_df['qty'].sum()
#     new_row = {'task_id':task_id, 'lot':'', 'wafer':'', 'site':'', 'insertion':str(get_insertion[0]), 'hbin':hbin_others, 'sbin':sbin_others, 'yield':sum_yield, 'offset':sum_offset, 'qty':sum_qty}
#     return new_row




# # here the code to calculate and put others into df
# def others_row(inputdf):
#     holding_df = pd.DataFrame(columns = ['task_id', 'lot', 'wafer', 'site', 'insertion', 'hbin', 'sbin', 'yield', 'offset', 'qty'])
#     lot_uniq = inputdf.lot.unique()
    
#     # do loop to go through each of the different lot
#     for i in range(len(lot_uniq)):
#         tempdf = inputdf.loc[inputdf['lot'] == lot_uniq[i]].reset_index(drop=True)
#         print(len(tempdf.columns))
#         if len(tempdf.columns) == 10:
#             wafer_uniq = tempdf.wafer.unique()
#             for j in range(len(wafer_uniq)):
#                 tempdf1 = tempdf.loc[tempdf['wafer'] == wafer_uniq[j]].reset_index(drop=True)
                
#                 if len(tempdf.columns) == 11:
#                     site_uniq = tempdf1.site.unique()
#                     for k in range(len(site_uniq)):
#                         tempdf2 = tempdf1.loc[tempdf1['site'] == site_uniq[k]].reset_index(drop=True)
#                         new_row=others_row_val(tempdf2)
#                         new_row['lot'] = lot_uniq[i]
#                         new_row['wafer'] = wafer_uniq[j]
#                         new_row['site'] = site_uniq[k]
#                         holding_df.append(new_row, ignore_index=True)
#                     break
                
#                 new_row=others_row_val(tempdf1)
#                 new_row['lot'] = lot_uniq[i]
#                 new_row['wafer'] = wafer_uniq[j]
#                 holding_df.append(new_row, ignore_index=True)
#             break
                                              
#         new_row=others_row_val(tempdf)
#         new_row['lot'] = lot_uniq[i]
#         holding_df = holding_df.append(new_row, ignore_index=True)
        
        
#     return holding_df#.drop(columns=['level_0', 'index'])




# # here to prep the lot data df into the lot, lotwafer, lotwafersite so that can do the others calculation
# masklot = lot_data.loc[lot_data['wafer'] == '']
# masklot.drop(columns=['wafer', 'site']).reset_index()

# maskwafer = lot_data.loc[lot_data['site'] == '']
# maskwafer = maskwafer.loc[maskwafer['wafer'] != '']
# maskwafer.drop(columns=['site']).reset_index()

# masksite = lot_data.loc[lot_data['site'] != ''].reset_index()

# after that can call the function with the mask df made earlier


# --- 4. Insert df to Oracle database ---



if df.empty != True:

    db = DatabaseConnection(config)
    engine = db.engine

    # open db connection
    connection = engine.connect()

    # insert into lot_data
        # convert all object types to varchar to insert data faster (faster performance)
    dtyp = {c:types.VARCHAR(lot_data[c].str.len().max())
           for c in lot_data.columns[lot_data.dtypes == 'object'].tolist()}

        # append the existing data with this data into db
    lot_data.to_sql("lot_data", engine, if_exists="append", dtype=dtyp, index=False)
    # lot_data.to_sql("lot_data", engine, if_exists='replace', dtype=dtyp, index=False)

        # make sure that lot_data has been inserted properly
    print(pd.read_sql_table("lot_data", connection))

    connection.close()




print('entire lot data script done')
















