import datetime as dt
from datetime import datetime
import random 
import pandas as pd 

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
#preprocess referral dates
#convert any day in any month to the first day of that month in REFERRAL data
#mutable obj integrity checked
#ref_copy passed by reference, changed directly
def convert_dates_ref(ref_copy, c_r): #use if c_r['date_format'] is 'y-m-d' or datetime
    ref_date_column = ref_copy[c_r['columns']['date_column']]
    for i in range(len(ref_date_column)):
        formatted_date = ref_date_column[i]
        new_formatted_date = datetime(formatted_date.year, formatted_date.month, c_r['day_to_evaluate'])
        ref_copy.at[i,c_r['columns']['date_column']] = new_formatted_date

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
'''
pd.to_Datetime(anything) = create datetime object
datetime.strptime(string) -> datetime object
datetime.strftime(datetime) -> string in different formats
'''


#give a 'ym' column, it will convert all of them to datetime
#mutable obj integrity checked
#updated mutable dataframe passed by reference
def ym_to_datetime(config, df): #ym to datetime
    label = config['columns']['date_column']
    df[label] = df[label].astype(str)
    for i in range(df[label].size):
        df[label][i] = df[label][i][:4] + '-' + df[label][i][4:] + '-01' #choosing the first month of the day
    df[label] = pd.to_datetime(df[label])
    #return df
    

#for evaluation
def drop_recent_referrals(c_p, c_r, c_e, ref_copy, pred_copy):
    start = c_e['eval_date'] - pd.DateOffset(months=c_e['recent_ref_window'])
    end = c_e['eval_date'] - pd.DateOffset(months=1) #ends the previous month of the current month
    mask = (ref_copy[c_r['columns']['date_column']]>=start) & (ref_copy[c_r['columns']['date_column']]<=end)
    
    #those IDs from prediction (0-500) that are not in list of those IDs that fall in daterange of all IDs(0-1000)
    idx = ~pred_copy[c_p['columns']['id_column']].isin(ref_copy.loc[mask][c_r['columns']['id_column']])
    pred_copy = pred_copy[idx]
    return pred_copy

#[0,3] -> datetime range in string
def window_to_range(c_e, window):
    
    if(window[0]<0):
        start = c_e['eval_date'] - pd.DateOffset(months=(-1)*window[0])
    else:
        start = c_e['eval_date'] + pd.DateOffset(months=window[0])

    end = c_e['eval_date'] + pd.DateOffset(months=window[1])
    
    start = start.strftime('%Y/%m')
    end = end.strftime('%Y/%m')
    
    win_range = start + '-' + end
    return win_range