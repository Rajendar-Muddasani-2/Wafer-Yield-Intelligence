#!/usr/bin/env python
# coding: utf-8

"""
lot_analysis.py - Fourth stage of ETL pipeline
Performs ML-based disposition analysis and generates retest recommendations
"""

import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline 
import cx_Oracle

from sqlalchemy import types, create_engine

# to create dir
import os
from os.path import join
from config_loader import Config
from database import DatabaseConnection

# to save models
from compress_pickle import load
import joblib

# for machine learning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# to calculate score of model
from sklearn import metrics
from scipy import interpolate
from scipy import stats
import skimage
from skimage import measure
from skimage.transform import radon
from skimage.transform import probabilistic_hough_line
from skimage import measure

# import tensorflow as tf
# import keras
# from keras import layers, Input, models
# from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import time

datapath = join('data', 'wafer')

import warnings
warnings.filterwarnings("ignore")


# --- 1. Inputs ---



# Load configuration from .env file
config = Config()

# select csv file to read
file_link = "new_records_only.csv"

# create output folders
    # make models folder if it doesn't exists to put files to insert to db
os.makedirs("./models", exist_ok = True)

models_folder = "./models/"

# trained ML model to use to predict sbin_run2
model = 'VC'

# files to retrieve from db
files_to_retrieve = [model + "_model.sav"] + ["training_header.csv"]

lot_analysis_cols = ["task_id", "lot", "wafer", "sbin", "insertion", "yield", "offset", "pf", "root_cause", "retest", "wafer_signature"]

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


# --- Read from database ---

# 2. Read latest task_id from lot_info, data from lot_data table and lot_stats table to find root cause



try:
    db = DatabaseConnection(config)
    engine = db.engine
    
    # open db connection
    connection = engine.connect()

    # read lot_data table
    existing_lot_data = pd.read_sql_table("lot_data", connection)
    print("Retrieved existing lot_data successfully.")
    
    # read lot_data table
    existing_lot_info = pd.read_sql_table("lot_info", connection)
    print("Retrieved existing lot_info successfully.")
    
    # read lot_stats table
    lot_stats = pd.read_sql_table("lot_stats", connection)
    print("Retrieved lot_stats successfully.")
    
    # get task id by max id
    task_id = existing_lot_info.task_id.max()
    
except Exception as e:
    print(e)
    
finally:
    connection.close()
    


# --- Convert to lot_analysis format ---

# Filter lot data to get the records of the latest task_id, lot and (SB0 or offset < 0)



# most recent records only
existing_lot_data = existing_lot_data[existing_lot_data["task_id"] == task_id]

# # lot level only
# lot_data = existing_lot_data[existing_lot_data["scope"].str.contains("_") == False]
# need to change to wafer level
lot_data = existing_lot_data[(existing_lot_data["wafer"].str.contains("None") == False) & (existing_lot_data["site"].isnull())]

# groupby lot, wafer, sbin, and so on...
lot_data = lot_data.groupby(['task_id', 'lot', 'wafer', 'sbin', 'insertion', 'yield', 'offset']).size().reset_index().rename(columns={0:'count'})
lot_data = lot_data.astype({'sbin': str, 'offset': float})
lot_data = lot_data.drop(columns=['count'])

# SB0 or offset < 0 
lot_analysis = lot_data[(lot_data["sbin"] == "0") | (lot_data["offset"] < 0)].reset_index(drop = True)  

lot_analysis

print('done filtering data to get records of latest task id and lot')


# Get pass/fail



def get_pf(row):
    pf = "pass"
    
    if row["offset"] < 0:
        pf = "fail"
    
    row["pf"] = pf
    
    return row




lot_analysis = lot_analysis.apply(get_pf, axis = "columns")
print('getting pass fail result')
lot_analysis


# Format root cause column



# not using this method
def format_root_cause(rc_list):
    rc_set = set()
    rc_str = ""
    
    rc_list.sort(key=len, reverse=True)
    
    for rc in rc_list:
        scope = rc.split("_")
        
        wafer = scope[1]
        
        if len(scope) == 3:
            rc_str += f"wafer{wafer}_site{scope[2]}, "
            rc_set.add(wafer)
        
        elif len(scope) == 2 and wafer not in rc_set:
                rc_str += f"wafer{wafer}, "
                
    return rc_str[:-2]


# Get the root cause site for wafer level



def format_site_root(in_df, row_lot, row_wafer, row_insertion, row_sbin):
    rc_str = ""
    # make sure to filter out, to get the same lot and wafer
    mask = in_df[(in_df['lot'] == row_lot) & (in_df['wafer'] == row_wafer)]
    
    # get total to calculate percentage of the particular site over total sbin. and this df is the existing lot data 
    tempdf = existing_lot_data[existing_lot_data["site"].isnull() == False].reset_index(drop=True) #get site level data
    tempdf = tempdf.loc[(tempdf["insertion"] == row_insertion)] #get correct insertion    
    tempdf = tempdf.loc[(tempdf['lot'] == row_lot) & (tempdf['wafer'] == row_wafer) & (tempdf['sbin'] == int(row_sbin))].reset_index() #get correct lot and wafer
    total = tempdf['qty'].sum() #calculate the total, so if want to show percentage, can use it
    
    # thus get the different site for the correct lot and wafer
    mask = mask.astype({'site': int})
    mask_list = mask['site'].values.tolist()
    mask_list.sort()
    
    for site in mask_list:
        tempdf_mask = tempdf.loc[(tempdf['site'] == str(site))]
        val = tempdf_mask['qty'].sum()

        # calculate the site qty over total wafer as percentage
        #val = (val/total) * 100 ---- if want to calculate percentage
        rc_str += f"site{site} {val}qty, "
        
    return rc_str[:-2]


# Get root cause



def get_root_cause(row): 
    
    if row["pf"] == "fail":
        
        root_cause = "unknown"
        
        # filter scope to give only wafer and site level in lot_stats
        #mask = lot_stats[lot_stats["scope"].isin(["wafer", "site"])]
        mask = lot_stats[lot_stats["scope"].isin(["site"])] # now change to wafer level, not lot. so changes
        
        #filter insertion and sbin in lot_stats
        mask = mask.loc[(mask["insertion"] == row["insertion"]) & (mask["sbin"] == str(row["sbin"]))]
        
        # if lot_stats does not have the particular insertion and sbin
        if mask.empty:
            root_cause = "new"
        else:
            # get wafer and site level for existing lot_data
            #lot_data = existing_lot_data[existing_lot_data["scope"].str.contains("_")] #.str.contains("None") == False
            lot_data = existing_lot_data[existing_lot_data["site"].isnull() == False].reset_index(drop=True) #get site level data, since entire table change to wafer level
            
            # offset < 0
            lot_data = lot_data.loc[lot_data["offset"] < 0]
            
            # filter to only have wafer and site relevant to row.lot
            #lot_data = lot_data[lot_data["scope"].str.contains(row.scope)] 
            lot_data = lot_data[(lot_data["wafer"].str.contains(row.wafer))] #made changes to site 
            
            #filter insertion and sbin from lot_data
            lot_data = lot_data.loc[(lot_data["insertion"] == row["insertion"]) & (lot_data["sbin"] == int(row["sbin"]))]
            
            if not lot_data.empty:
                #root_cause = format_root_cause(list(lot_data.scope))
                root_cause = format_site_root(lot_data, row['lot'], row['wafer'], row['insertion'], row['sbin'])
            
        if root_cause == '':
            root_cause = 'NA'
            
        row["root_cause"] = root_cause

    else:
        row["root_cause"] = 'NA'
        # format to get consistent column name order
        temp_col = ['task_id', 'lot', 'wafer', 'sbin', 'insertion', 'yield', 'offset', 'pf', 'root_cause']
        row = row[temp_col]
    return row




lot_analysis = lot_analysis.apply(get_root_cause, axis = "columns")
print('getting root cause \n')
print(lot_analysis)


# Get retest

# Read data and rename accordingly for ML



print('reading new records csv file')
df = pd.read_csv(file_link, sep=",", low_memory=False)

if df.empty:
    print('df is empty, quitting script')
    quit()

# get product and tp_ver to select training header and trained ML models
product = df["producttype"][0]
tp_ver = df["testprgrev"][0].replace(".", "")

# rename sbin to sbin_run1
df = df.rename(columns = {'sbin':'sbin_run1'})

print('renaming column name')

df


# 3. Get training header and trained model from lot_ml table and save to local disk



# get model from db
connection = cx_Oracle.connect(username, password, tns, encoding="UTF-8")
cursor = connection.cursor()

print('trying to get prediction retest ml model')

try:
    
    for file in files_to_retrieve:
        # execute sql statement
        cursor.execute("select * from lot_ml where typ = :fileVer", fileVer = product + "_" + tp_ver + "_" + file)
        
        
        # fetch one result
        obj, name, date_time1 = cursor.fetchone()

        f = open(models_folder + name, "wb")
        f.write(obj.read())
        f.close()
        
    print('ml model used is for product ' + product + ' and for tp ver ' + tp_ver)
except Exception:
    
    # execute sql statement to get the latest date
    cursor.execute("select s1.* from lot_ml s1 inner join(select max(datetime) datetime, typ from lot_ml group by typ) s2 on s1.datetime = s2.datetime and s1.typ = s2.typ where s1.typ NOT LIKE 'PR%'") 
    
    # save the actual product & tp_ver into temp variable first
    temp_prod, temp_tpver = product, tp_ver
                   
    # fetch one result
    obj, name, date_time2 = cursor.fetchone()

    # to get the name and date
    # eg 'M3990B00013_P0L_VC_model.sav'
    first_ind_end = name.index("_")
    product = name [0 : first_ind_end]
    
    sec_ind_end = name.index("_", first_ind_end + 1)
    tp_ver = name [first_ind_end + 1 : sec_ind_end]               
    
    for file in files_to_retrieve:                   
        # execute sql statement
        cursor.execute("select * from lot_ml where typ = :fileVer", fileVer = product + "_" + tp_ver + "_" + file)

        
        # fetch one result
        obj, name, date_time3 = cursor.fetchone()
                   
        f = open(models_folder + name, "wb")
        f.write(obj.read())
        f.close()
    
    print('ml model used is the latest trained model, product ' + product + ' and tp ver ' + tp_ver)
finally:
    connection.close()


# Filter data to only have sbin<>0




print('filtering data that have failed the test')
df = df.loc[df.sbin_run1 != 0].reset_index(drop = True)  
df


# Rearrange df header sequence



print('rearranging df header sequence')
header = pd.read_csv(models_folder + f"{product}_{tp_ver}_training_header.csv", sep =";", low_memory=False, encoding = "ISO-8859-1")

# convert df to list
header = header["column"].to_list()

# get a copy of set(header) 
header_copy = set(header.copy())

# values that are both in def and stdf
in_both = set([n for n in df.columns if n in header_copy])

# values that are in header but not in df
remaining = list(header_copy - in_both)

# add a column for the remaining values
for each in remaining:
    df[each] = 0
    
# rearrange stdf to header
df_arranged = df[header]

df_arranged


# Replace all NaN with 0 in X_test



X_test = df_arranged.copy()

print('replacing nan with zero')

# to convert all values to numeric, the data contain some weird data, so 'force' it to change if otherwise
for col in X_test.columns:
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

# replace all NaN with 0
X_test = X_test.fillna(0)

# convert all values to numeric
X_test = X_test.apply(pd.to_numeric)


X_test


# Scaling Features using Z-score



def apply_zscore(df, replaceNaN = 0):
    df_normalized = (df - df.mean()) / df.std()
    
    # replace NaN with 0
    df_normalized = df_normalized.fillna(replaceNaN)
    
    return df_normalized




df_normalized = apply_zscore(X_test)
df_normalized
print('normalizing the data')


# Apply trained model



def predict_sbin_run2(X, model):

    # load trained model
    loaded_model = load(open(models_folder + f"{product}_{tp_ver}_{model}_model.sav", 'rb'), compression="gzip", set_default_extension=False)

    # score is basically predict and then compare the predicted with the actual labels
    y_pred = loaded_model.predict(X)

    # add predicted labels to X_test
    df["predicted_sbin_run2"] = y_pred
    
    # convert to string
    df["predicted_sbin_run2"] = df["predicted_sbin_run2"].astype(str)
    
    print("Successfully predicted sbin_run2.")




# input model to be tested
predict_sbin_run2(df_normalized, model)


# df.to_csv("review.csv", index = False)
df


# Get recover quantity



def get_wafers_only(rc):
    rc_list = rc.split(", ")

    result = set()
    for each in rc_list:
        wafer_site = each.split("_")
        wafer = wafer_site[0][5:]

        result.add(wafer)

    return list(result)




def get_retest(row, df):
    result = ''
    
    # skip to next row if sbin = 0 or pf = pass
    if row.sbin != "0" or row.pf != "pass":
        # filter df by insertion, sbin, scope 
        cols = ["measstep", "sbin_run1", "lot", "wafer"]
        criteria = {"measstep":[row.insertion], "sbin_run1":[row.sbin], "lot":[row.lot], "wafer":[row.wafer]}
        df = df.astype({'sbin_run1': str, 'wafer': str})
        mask = df[df[cols].isin(criteria).all(axis=1)].reset_index()
        
#         # if there is root cause
#         if not pd.isnull(row.root_cause):
#             # filter by wafer
#             wafers = get_wafers_only(row.root_cause)
#             mask = mask.loc[mask['wafer'].isin(wafers)]
        
        # calculate predict_sbin_run2 = 0 / all records of sbin_run1 <> 0
            # since now wafer level, and don't need the percentage, so comment out
        qty_pass = len(mask[mask["predicted_sbin_run2"] == "0"].index)
        result = str(qty_pass)
        
#         if qty_pass != 0 or len(mask.index) != 0:
#             #percent_pass = (qty_pass / len(mask.index)) * 100
            
#             result = str(qty_pass) #+ ", " + str(percent_pass) + "%"
#         else:
#             result = str(qty_pass) #+ ", 0%" 

    # insert retest col
    row["retest"] = result
    
    return row




print('getting recover quantity and percentage')

lot_analysis = lot_analysis.apply(get_retest, axis = 'columns', df = df)

lot_analysis


# From retest prediction, get predicted value and add into sbin 0 



print('calculating predicted passed retest')

retest_total = 0
passed_total = 0

# take a copy of the df
temp_lot_ana = lot_analysis.copy()
# take out only those fail
temp_lot_ana = temp_lot_ana.loc[(temp_lot_ana['pf'] == 'fail')].reset_index(drop=True)
# get the unique lots that failed
failed_lots = temp_lot_ana.groupby(['lot','wafer']).size().reset_index()

# to loop through the different failed lots
for index, row in failed_lots.iterrows():
    tempdf_lot_a = temp_lot_ana.copy()
    tempdf_lot_a = tempdf_lot_a.loc[(tempdf_lot_a['lot'] == row['lot'])]
    tempdf_lot_a = tempdf_lot_a.loc[(tempdf_lot_a['wafer'] == row['wafer'])].reset_index()
    # change retest type to int just in case
    tempdf_lot_a = tempdf_lot_a.astype({'retest': int})
    retest_total = tempdf_lot_a['retest'].sum()
    
    # from the lot data, get wafer level data
    lot_data = existing_lot_data[existing_lot_data["site"].isnull() == True].reset_index(drop=True)
    lot_data = lot_data[(lot_data["wafer"] == row['wafer'])].reset_index(drop=True)
        
    # get passed wafer qty
    lot_data = lot_data.loc[lot_data["sbin"] == 0]
    passed_total = lot_data['qty'].sum()
    
    # get the index of the correct lot, wafer, and sbin by filtering
    mask_index = lot_analysis[(lot_analysis['lot'] == row['lot']) & (lot_analysis['wafer'] == row['wafer']) & (lot_analysis['sbin'] == '0')].index.values
    # from the df at the specific location, insert the value in
    lot_analysis.at[mask_index, 'retest'] = passed_total + retest_total
    
lot_analysis = lot_analysis.astype({'retest': str})
lot_analysis


# Plot the wafer map



print('preparing the wafer map info')

df = pd.read_csv(file_link, sep=",", low_memory=False)

df = df[["lot", "wafer", "sbin", "site_no", "x", "y"]]

df




# def assign_value(wafer, x, y):
#     mask = df[(df["wafer"] == wafer) & (df["x"] == x) & (df["y"] == y)].reset_index()
#     try:
#         #if in root cause
#         if lot_analysis.root_cause.str.contains(f'wafer{wafer}_site{mask.site_no[0]}', regex=False).any(axis=None):
#             return 2
        
#         #if not in root cause
#         else:
#             return 1
#     except:
#         # padding
#         return 0
def ass_value(lot, wafer, sbin, x, y):
    mask1 = df[(df["lot"] == str(lot)) & (df["wafer"] == wafer) & (df["sbin"] == sbin) & (df["x"] == x) & (df["y"] == y)].reset_index()
    #print(mask1.sbin)
    #print(f'lot is: {lot}, wafer is: {wafer}, x val is: {x}, y val is: {y}')
    if not mask1.empty:
        if (mask1.sbin).item() == 0: # check if sbin pass. if yes pass, is 1
            return 1
        else: # if fail, is 2
            return 2 ##include another one that states the bin that caused the failure
    else: #0 for the padding
        return 0




# get a partial copy of the lot analysis df to get the wafer map coordinates/data
sub_df = lot_analysis[['lot', 'wafer', 'sbin', 'insertion', 'yield', 'offset']].copy()
sub_df = sub_df.astype({'wafer': int, 'sbin': int, 'yield':float, 'offset':float})
df = df.astype({'wafer': int, 'x': int, 'y': int, 'sbin': int})
# to set the lower and upper limit... shifted it into the loop, so that more accurate
# min_x, max_x = df.x.min(), df.x.max()
# min_y, max_y = df.y.min(), df.y.max()

twafers = {}
temp_lst = []
tempcou = 0
anotherlist = []

print('calculating wafermap coordinates. may take awhile...')

# loop through each row from the df to add the x and y coordinates
for index, row in sub_df.iterrows():
    wafer_xy = []
    # shifted the setting of the max and min for the xy into the loop so that the printed array will be smaller, 
    # without padding, more accurate
    inner_df = df[(df["lot"] == str(row['lot'])) & (df["wafer"] == row['wafer'])].reset_index()
    min_x, max_x = inner_df.x.min(), inner_df.x.max()
    min_y, max_y = inner_df.y.min(), inner_df.y.max()
    
    if(np.isnan(min_x)):
        min_x = 0
    if(np.isnan(max_x)):
        max_x = 0
    if(np.isnan(min_y)):
        min_y = 0
    if(np.isnan(max_y)):
        max_y = 0
        
    for y in range(min_y, max_y + 1):
        ys = []
        for x in range(min_x, max_x + 1):
            ys.append(ass_value(row['lot'], row['wafer'], row['sbin'], x, y))

        wafer_xy.append(ys)

    #twafers[str(lot) + '_' + str(wafer)] = wafer_xy
    some_list = []
    some_list.append(row['lot'])
    some_list.append(row['wafer'])
    some_list.append(row['sbin'])
    some_list.append(row['insertion'])
    some_list.append(row['yield'])
    some_list.append(row['offset'])
    anotherlist.append(np.asarray(wafer_xy,dtype='uint8'))
    temp_lst.append(some_list)
    inner_df=None
    
    
tempdff = pd.DataFrame(temp_lst, columns = ['lot', 'wafer', 'sbin', 'insertion', 'yield', 'offset'])

#twafers
tempdff['waferMap'] = anotherlist

print('done calculating wafermap array')




# min_wafer, max_wafer = df.wafer.min(), df.wafer.max()
# min_x, max_x = df.x.min(), df.x.max()
# min_y, max_y = df.y.min(), df.y.max()

# wafers = {}
# for wafer in range(min_wafer, max_wafer + 1):
#     wafer_xy = []
#     for y in range(min_y, max_y + 1):
#         ys = []
#         for x in range(min_x, max_x + 1):
#             ys.append(assign_value(wafer, x, y))
            
#         wafer_xy.append(ys)
        
#     wafers[wafer] = wafer_xy
  
# wafers



# # here from a1g data, get the np array of the xy coordinates
# df = df.astype({'wafer': int, 'x': int, 'y': int, 'sbin': int})
# min_wafer, max_wafer = df.wafer.min(), df.wafer.max()
# min_x, max_x = df.x.min(), df.x.max()
# min_y, max_y = df.y.min(), df.y.max()
# min_sbin, max_sbin = df.sbin.min(), df.sbin.max()
# lot_num = df.lot.unique()

# twafers = {}
# temp_lst = []
# tempcou = 0
# anotherlist = []

# for lot in lot_num:
#     for wafer in range(min_wafer, max_wafer + 1):
#         wafer_xy = []
#         for sbin in range(min_sbin, max_sbin + 1):

#             for y in range(min_y, max_y + 1):
#                 ys = []
#                 for x in range(min_x, max_x + 1):
#                     ys.append(ass_value(lot, wafer, sbin, x, y))

#                 wafer_xy.append(ys)

#             #twafers[str(lot) + '_' + str(wafer)] = wafer_xy
#             some_list = []
#             some_list.append(lot)
#             some_list.append(wafer)
#             some_list.append(sbin)
#             anotherlist.append(np.asarray(wafer_xy,dtype='uint8'))
#             temp_lst.append(some_list)
#     tempcou = tempcou + 1
# tempdff = pd.DataFrame(temp_lst, columns = ['lot', 'wafer', 'sbin'])

# #twafers
# tempdff['waferMap'] = anotherlist
# print(len(anotherlist))
# print(min_y, max_y, min_x, max_x)
# print(type(tempdff.lot[0]))




#print('plotting wafer maps')

# for wafer in wafers:
#     print(wafer)
#     plt.imshow(wafers[wafer])
#     plt.show()


# counter_for_plt_loop = 0
# for i in tempdff.waferMap:
#     print(f"lot {tempdff.lot[counter_for_plt_loop]}")
#     print(f"wafer {tempdff.wafer[counter_for_plt_loop]}")
#     print(f"sbin number {tempdff.sbin[counter_for_plt_loop]}")
#     plt.imshow(i)
#     plt.show()
#     counter_for_plt_loop = counter_for_plt_loop + 1




#lot_analysis




#sub_df




#tempdff




# get the supervised trained model from Oracle DB
try:
    # get model from db
    connection = cx_Oracle.connect(username, password, tns, encoding="UTF-8")
    cursor = connection.cursor()

    print('trying to get pattern recognition model')

    # get the first pre trained model, gradient boost
    cursor.execute("select * from lot_ml where typ = :fileVer", fileVer = 'PR_GB_model_9cat.sav')

    # fetch GB result
    obj, nameGB, date_time1 = cursor.fetchone()
    GB = open("./models/" + nameGB, "wb")
    GB.write(obj.read())
    GB.close()
    
    # get the second pre trained model, multi layer perceptron
    cursor.execute("select * from lot_ml where typ = :fileVer", fileVer = 'PR_MLP_model_9cat.sav')

    # fetch MLP result
    obj, nameMLP, date_time1 = cursor.fetchone()
    MLP = open("./models/" + nameMLP, "wb")
    MLP.write(obj.read())
    MLP.close()
    
    # get the last pre trained model, voting ensemble
    cursor.execute("select * from lot_ml where typ = :fileVer", fileVer = 'PR_VE_model_9cat.sav')

    # fetch VE result
    obj, nameVE, date_time1 = cursor.fetchone()
    VE = open("./models/" + nameVE, "wb")
    VE.write(obj.read())
    VE.close()

except Exception as e:
    print(e)
    
finally:
    connection.close()
    print('Pattern recognition model taken from Oracle DB')
    


# <h2>The few cells below, is to prep the waferMap numpy array so that can do pattern recognition.
# For more info, can look at the notebook 'wafer pattern recognition...' from the same dir</h2>



# calculate the density
def cal_den(x):
    return 100*(np.sum(x==2)/np.size(x))  

# wafer divided into 13 region, split it up accordingly
def find_regions(x):
    rows=np.size(x,axis=0)
    cols=np.size(x,axis=1)
    if (rows == 0 or cols == 0):
        return 0
    else:
        if (rows < 5):
            ind1=np.arange(0,rows,1)
        else:
            ind1=np.arange(0,rows,rows//5)
        if(cols < 5):
            ind2=np.arange(0,cols,1)
        else:    
            ind2=np.arange(0,cols,cols//5)
        reg1=x[ind1[0]:ind1[1],:]
        reg3=x[ind1[4]:,:]
        reg4=x[:,ind2[0]:ind2[1]]
        reg2=x[:,ind2[4]:]

        reg5=x[ind1[1]:ind1[2],ind2[1]:ind2[2]]
        reg6=x[ind1[1]:ind1[2],ind2[2]:ind2[3]]
        reg7=x[ind1[1]:ind1[2],ind2[3]:ind2[4]]
        reg8=x[ind1[2]:ind1[3],ind2[1]:ind2[2]]
        reg9=x[ind1[2]:ind1[3],ind2[2]:ind2[3]]
        reg10=x[ind1[2]:ind1[3],ind2[3]:ind2[4]]
        reg11=x[ind1[3]:ind1[4],ind2[1]:ind2[2]]
        reg12=x[ind1[3]:ind1[4],ind2[2]:ind2[3]]
        reg13=x[ind1[3]:ind1[4],ind2[3]:ind2[4]]

        fea_reg_den = []
        fea_reg_den = [cal_den(reg1),cal_den(reg2),cal_den(reg3),cal_den(reg4),cal_den(reg5),cal_den(reg6),cal_den(reg7),cal_den(reg8),cal_den(reg9),cal_den(reg10),cal_den(reg11),cal_den(reg12),cal_den(reg13)]
        return fea_reg_den




def change_val(img):
    img[img==1] =0  
    return img




# cubic interpolated mean calculation
def cubic_inter_mean(img):
    theta = np.linspace(0., 180., max(img.shape), endpoint=False)
    sinogram = radon(img, theta=theta)
    xMean_Row = np.mean(sinogram, axis = 1)
    x = np.linspace(1, xMean_Row.size, xMean_Row.size)
    y = xMean_Row
    f = interpolate.interp1d(x, y, kind = 'cubic')
    xnew = np.linspace(1, xMean_Row.size, 20)
    ynew = f(xnew)/100   # use interpolation function returned by `interp1d`
    return ynew

# cubic interpolated standard deviation calculation
def cubic_inter_std(img):
    theta = np.linspace(0., 180., max(img.shape), endpoint=False)
    sinogram = radon(img, theta=theta)
    xStd_Row = np.std(sinogram, axis=1)
    x = np.linspace(1, xStd_Row.size, xStd_Row.size)
    y = xStd_Row
    f = interpolate.interp1d(x, y, kind = 'cubic')
    xnew = np.linspace(1, xStd_Row.size, 20)
    ynew = f(xnew)/100   # use interpolation function returned by `interp1d`
    return ynew  




# some other calculation
def cal_dist(img,x,y):
    dim0=np.size(img,axis=0)    
    dim1=np.size(img,axis=1)
    dist = np.sqrt((x-dim0/2)**2+(y-dim1/2)**2)
    return dist  

# some other calculation
def fea_geom(img):
    norm_area=img.shape[0]*img.shape[1]
    norm_perimeter=np.sqrt((img.shape[0])**2+(img.shape[1])**2)
    
    #img_labels = measure.label(img, neighbors=4, connectivity=1, background=0)
    img_labels = measure.label(img, connectivity=1, background=0)

    if img_labels.max()==0:
        img_labels[img_labels==0]=1
        no_region = 0
    else:
        info_region = stats.mode(img_labels[img_labels>0], axis = None)
        no_region = info_region[0][0]-1       
    
    prop = measure.regionprops(img_labels)
    prop_area = prop[no_region].area/norm_area
    prop_perimeter = prop[no_region].perimeter/norm_perimeter 
    
    prop_cent = prop[no_region].local_centroid 
    prop_cent = cal_dist(img,prop_cent[0],prop_cent[1])
    
    prop_majaxis = prop[no_region].major_axis_length/norm_perimeter 
    prop_minaxis = prop[no_region].minor_axis_length/norm_perimeter  
    prop_ecc = prop[no_region].eccentricity  
    prop_solidity = prop[no_region].solidity  
    
    return prop_area,prop_perimeter,prop_majaxis,prop_minaxis,prop_ecc,prop_solidity




# backup the df just in case
temp_df_bkup = tempdff.copy()
# drop those sbin = 0, so that lesser row to do prediction
tempdff = tempdff.loc[tempdff.sbin != 0].reset_index(drop=True)
temp_df_copy = tempdff.copy()

# all the calculation
print('calculating wafermap stats')
temp_df_copy['fea_reg']=temp_df_copy.waferMap.apply(find_regions)
temp_df_copy['new_waferMap'] = temp_df_copy.waferMap.apply(change_val)

temp_df_copy['fea_cub_mean'] = temp_df_copy.waferMap.apply(cubic_inter_mean)
temp_df_copy['fea_cub_std'] = temp_df_copy.waferMap.apply(cubic_inter_std)

temp_df_copy['fea_geom'] = temp_df_copy.waferMap.apply(fea_geom)




# more calculation
tempdf_all=temp_df_copy.copy()
e=[tempdf_all.fea_reg[i] for i in range(tempdf_all.shape[0])] 
f=[tempdf_all.fea_cub_mean[i] for i in range(tempdf_all.shape[0])] 
g=[tempdf_all.fea_cub_std[i] for i in range(tempdf_all.shape[0])] 
h=[tempdf_all.fea_geom[i] for i in range(tempdf_all.shape[0])]
if not tempdf_all.empty:
    print('calculating features')
    tempfea_all = np.concatenate((np.array(e),np.array(f),np.array(g),np.array(h)),axis=1) 
else:
    print('df empty for sbin not equal zero, no feature to calculate')




# Import the model we are using
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

#start_time = time.time()

# load trained model
gbwl = load(open("./models/PR_GB_model_9cat.sav", 'rb'), compression="gzip", set_default_extension=False)
mlpwl = load(open("./models/PR_MLP_model_9cat.sav", 'rb'), compression="gzip", set_default_extension=False)
loaded_model_VE = load(open("./models/PR_VE_model_9cat.sav", 'rb'), compression="gzip", set_default_extension=False)

# score is basically predict and then compare the predicted with the actual labels
if not tempdf_all.empty:
    print('predicting wafer pattern...')
    y_pred = loaded_model_VE.predict(tempfea_all)

    # add predicted labels to df
    temp_df_copy["predicted_pattern"] = y_pred

    # renaming from numbers to the pattern name
    mapping_type_to_name={0:'Center',1:'Donut',2:'Edge-Local',3:'Edge-Ring',4:'Local',5:'Random',6:'Scratch',7:'Near-full',8:'none'}
    temp_df_copy=temp_df_copy.replace({'predicted_pattern':mapping_type_to_name})

    # convert to string
    temp_df_copy["predicted_pattern"] = temp_df_copy["predicted_pattern"].astype(str)

    print("Successfully predicted pattern")

else:
    print('df empty, nothing to predict')




# this df, is the filtered sbin=0 rows. so fill them with na for the prediction pattern
atempdf = temp_df_bkup.loc[temp_df_bkup.sbin == 0].reset_index(drop=True)
atempdf["predicted_pattern"] = ''

# combine the sbin!= 0 and == 0 together and sort, so that can easily add onto the lot analysis df  
temp_agn_df = temp_df_copy.append(atempdf, ignore_index=True)
temp_agn_df = temp_agn_df.astype({'lot': str,'wafer': int, 'sbin': int, 'insertion': str, 'yield':float, 'offset':float})
temp_agn_df = temp_agn_df.sort_values(by=['lot', 'wafer', 'sbin', 'insertion', 'yield', 'offset']).reset_index(drop=True)




temp_agn_df




# sorting the df so that can easily add the wafer sig onto this df
lot_analysis = lot_analysis.astype({'lot': str,'wafer': int, 'sbin': int, 'insertion': str, 'yield':float, 'offset':float})
lot_analysis = lot_analysis.sort_values(by=['lot', 'wafer', 'sbin', 'insertion', 'yield', 'offset']).reset_index(drop=True)




# just add the wafer sig col to the lot ana df
lot_analysis['wafer_signature'] = temp_agn_df['predicted_pattern']




lot_analysis


# Finalise lot_analysis



# insert wafer_signature col
#lot_analysis["wafer_signature"] = np.nan
print('formating lot analysis table')
# reorganise to desired format
lot_analysis = lot_analysis[lot_analysis_cols]

# make all headers lowercase
lot_analysis.columns = map(str.lower, lot_analysis.columns)

# lot_analysis.to_csv("lot_analysis.csv")
lot_analysis


# --- Insert df to Oracle database ---



db = DatabaseConnection(config)
engine = db.engine

# open db connection
connection = engine.connect()
print('inserting lot analysis into Oracle DB')
# insert into lot_analysis
    # convert all object types to varchar to insert data faster (faster performance)
dtyp = {c:types.VARCHAR(lot_analysis[c].str.len().max())
       for c in lot_analysis.columns[lot_analysis.dtypes == 'object'].tolist()}

    # append the existing data with this data into db
lot_analysis.to_sql("lot_analysis", engine, if_exists='append', dtype=dtyp, index=False)
# lot_analysis.to_sql("lot_analysis", engine, if_exists='replace', dtype=dtyp, index=False)

    # make sure that lot_analysis has been inserted properly
print(pd.read_sql_table("lot_analysis", connection))

connection.close()




print('done inserting lot analysis table into Oracle DB')






