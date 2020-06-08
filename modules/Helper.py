import datetime as dt
from datetime import datetime
import random
import pandas as pd
import yaml
from IPython.core.display import display
import pickle
import s3fs
import pyarrow.parquet as pq
import Aws
import os 

#for generation
#generate random date between a range
#input in datetime.date format
#mutable obj integrity checked
def generate_random_date(start_date, end_date, iteration, date_format):
    date_list = []
    seed = 0
    for i in range(iteration):
        s_date = datetime.strptime(start_date, '%Y-%m-%d')
        e_date = datetime.strptime(end_date, '%Y-%m-%d')
        time_between_dates = e_date - s_date
        days_between_dates = time_between_dates.days
        random.seed(seed)
        random_number_of_days = random.randrange(days_between_dates)
        random_date = s_date + dt.timedelta(days=random_number_of_days) #this is datetime object
        if date_format=='y-m-d':
            random_date = random_date.strftime('%Y-%m-%d') #string format output
        elif date_format=='ym':
            random_date = int(random_date.strftime("%Y%m"))
        date_list.append(random_date)
        seed += 1

    return date_list

#for evaluation
#convert top k probability of a target to 1, O/W to 0 mapping
#target was a pd.series, kept intact
#mutable obj integrity checked
def prob_to_bin(target, k):
    target_bin = target.copy()
    target_bin[target_bin.index] = 0
    idx = target.nlargest(k).index
    target_bin[idx] = 1

    return target_bin
#return local variable, it does not get lost, stays in heap as long as there is a reference to it. O/W garbage collector.

#for evaluation
def drop_recent_referrals(c_p, c_r, c_e, eval_date, ref_copy, pred_copy):
    start = datetime.strptime(eval_date, '%Y%m') - pd.DateOffset(months=c_e['recent_ref_window'])
    end =   datetime.strptime(eval_date, '%Y%m') - pd.DateOffset(months=1) #ends the previous month of the current month
    mask = (pd.to_datetime(ref_copy[c_r['columns']['date_column']], format='%Y%m')>=start) & (pd.to_datetime(ref_copy[c_r['columns']['date_column']], format='%Y%m')<=end)

    #those IDs from prediction (0-500) that are not in list of those IDs that fall in daterange of all IDs(0-1000)
    idx = ~pred_copy[c_p['columns']['id_column']].isin(ref_copy.loc[mask][c_r['columns']['id_column']])
    pred_copy = pred_copy[idx]
    return pred_copy

#[0,3] -> datetime range in string
def window_to_range(c_e, window):

    if(window[0]<0):
        start = datetime.strptime(c_e['eval_date'], '%Y%m') - pd.DateOffset(months=(-1)*window[0])
    else:
        start = datetime.strptime(c_e['eval_date'], '%Y%m') + pd.DateOffset(months=window[0])
        
    end = datetime.strptime(c_e['eval_date'], '%Y%m') + pd.DateOffset(months=window[1])

    start = datetime.strftime(start, '%Y%m')
    end = datetime.strftime(end, '%Y%m')

    win_range = start + '-' + end
    return win_range

def load_yaml(file):
    with open(file) as f:
        return yaml.safe_load(f)

def load_configuration(config_file, prediction_config_files):
    config=load_yaml(config_file)
    c_r = config['c_r']  #Details on the referral file
    c_e = config['c_e']  #Details on the experiment settings (evaluation date, windows, etc.)
    c_gen = config['c_gen']  #Configuration for aws
    c_aws = config['c_aws']  #Configuration for aws
    c_visual = config['c_visual']
    #c_e['eval_date'] = pd.to_datetime(c_e['eval_date'])
    c_p=[]
    for file in prediction_config_files:
        config=load_yaml(file)
        c_p.append(config['c_p'])

    return c_r, c_e, c_gen, c_aws, c_visual, c_p

#returns a df
def read_file(directory, file, file_format, aws=False, bucket=None, temp_dir='../data/tmp/', filters=None):
    #need to handle 4 cases of local/s3/csv/parquet
         
    if file_format in ['csv', 'pickle']:
        if aws:
            df= Aws.load_from_aws(bucket=bucket, directory=directory, file=file)
        else:
            assert (os.path.exists(directory+file)), directory + file + " does not exist"
            df= pd.read_csv(directory+file)
        return df
    
    elif file_format == 'parquet':
        if aws:
            fs=s3fs.S3FileSystem()
            assert fs.exists('s3://'+bucket+directory+file), 's3://'+bucket+directory+file+" does not exist"
            uri = 's3://'+bucket+directory+file
        else:
            fs=None
            assert (os.path.exists(directory+file)), directory + file + " does not exist"
            uri=directory+file
        dataset=pq.ParquetDataset(uri, fs, filters=filters)
        table=dataset.read()
        df = table.to_pandas()
        del table
        return df

#get the min and max date in prediction file
def date_range(c_p, prediction):
    date_col = prediction[c_p['columns']['date_column']]
    min_date = min(date_col)
    max_date = max(date_col)
    print("Min Date in Prediction: ",min_date)
    print("Max Date in Prediction: ",max_date)

#overlap set check
def overlap_set_check(c_p, c_r, referral, prediction):
    r=set(referral[c_r['columns']['id_column']].unique())
    p=set(prediction[c_p['columns']['id_column']].unique())
    #print("Number of Unique IDs in Referral:",len(r))
    #print("Number of Unique IDs in Prediction:",len(p))
    result=p.intersection(r)
    assert result!=0, "No overlap between Referral and Prediction"
    #print("Number of Intersected IDs:", len(result))

#boolean assert
#check if all columns of column_dict from config exists in dataframe(referral/prediction)
def column_exists(dataframe, column_dict):
    df_columns = list(dataframe.columns)
    config_columns = list(column_dict.values())
    #print("Columns from dataframe (referral/prediction):", df_columns)
    #print("Columns from config(c_r/c_p):", config_columns)
    check =  all(item in df_columns for item in config_columns)
    return check

#returns a list of unique MYRs ascending order
def eval_date_extract(c_p, prediction):
    date_list = sorted(prediction[c_p['columns']['date_column']].unique())
    return [str(i) for i in date_list] 
