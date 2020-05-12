#Set config.
import sys, datetime, os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, accuracy_score, recall_score, balanced_accuracy_score, f1_score, roc_auc_score, log_loss, roc_curve
import itertools

def preprocess_referrals(c_r, drop_duplicates=True):
    """
    Preprocess referral data, aggregating reason codes, and referrals.
    """
    #ref (dataframe) = "/data/referrals/test.csv" - ['person_id', 'date', 'class'] - 23 referrals
    ref=pd.read_csv(c_r['dir']+c_r['file'])

    '''Translate Dates to the datetime format and then to int in two different columns'''
    #converts the ref['date'] column to datetime format and puts the converted values to a new column ref['datetime']
    #ref['date'] = '1/1/2017' -> ref['datetime'] = '2017-01-01'
    ref['datetime']=pd.to_datetime(ref[c_r['date_col_in']], format= c_r['date_for_in'])
    #new column yyyymm (int) = convert datetime to an int
    #2017-01-01 -> 201701
    ref[c_r['date_col_out']]=ref['datetime'].dt.strftime(c_r['date_for_out']).astype(int)

    ''' Get all the unique value from ref['class'] column and map 1...n for each value, zip in translate dictionary '''
    #This creates a class number for each reason code
    trans=ref[c_r['reason_col']].unique()
    translate=dict(zip(ref[c_r['reason_col']].unique(),[x for x in range(1,len(trans)+1)]))

    ''' Select important columns ref['person_id', 'yyyymm', 'class', 'datetime']'''
    #Select only the important columns.
    ref=ref.loc[:,[c_r['per_col'],c_r['date_col_out'], c_r['reason_col'],'datetime']]

    '''This creates a column with a label consistent with translation.'''
    #ref['person_id', 'yyyymm', 'class', 'datetime', 'label']
    ref['label']=ref[c_r['reason_col']].map(lambda x: translate[x])
    #creates one column for each different unique class for all samples and put 1 to the column to which that sample belong
    #if ref['label'] = 1, lab = [1,0,0]; if ref['label'] = 2, lab = [0,1,0] (all values of lab are in different columns)
    lab_dum=pd.get_dummies(ref['label'],prefix='lab')
    #append the lab_x columns to the end of ref
    #ref['person_id', 'yyyymm', 'class', 'datetime', 'label', 'lab_1', 'lab_2', 'lab_3']
    ref = pd.concat([ref, lab_dum], axis=1);

    '''This sums up the referrals.'''
    #rearrange ref using 'person_id' and 'yyyymm' as 1st and 2nd index, and then sum
    #basically tries -> if one patient in same month gets multiple referrals (How plausible is this??)
    ref = pd.pivot_table(ref, values=lab_dum.columns, index=[c_r['per_col'],c_r['date_col_out']], aggfunc=np.sum)

    #This sums the total number of referrals.
    #ref['ref_m']=ref.sum(axis=1)

    #This creates a binary variable for referial.
    # ???????? Why this Binary Variable column ???????
    ref['ref']=1

    if drop_duplicates==True:
        #get those columns that have 'lab_' as prefix
        cols=list(ref.columns[ref.columns.str[0:len('lab_')]=='lab_'])

        #inside those cols
        for c in cols:
            
            #if any sample has 1 or more than 1 for (person_id, yyyymm) index, then translate it to 1 
            #removing duplicates multiple referrals in one month for one person boils down to 1 referral
            ref[c]=ref[c].map(lambda z: 1 if z>=1 else 0)

    #reset the index
    ref.reset_index(inplace=True)

    #ref = ['person_id', 'yyyymm', 'lab_1', 'lab_2', 'lab_3']
    #translate = {'diabetes': 1, 'liver': 2, 'pnemonia': 3}
    return ref, translate

def score_times(c_p, c_r, c_e, ref=pd.DataFrame(), pred=pd.DataFrame() ):
    """
    c_p= The configruation for a set of predictions
    c_r= The configuration a set of referral data
    c_e= The configuration for an experiment.
    ref= The referral data dataframe
    pred= The prediction dataframe.
    """
    #Load the reference dataframe if not passed.
    if ref.empty == True:
        print("Loading reference dataframe..", c_r['file'])
        ref, trans = preprocess_referrals(c_r)

    else:
        print("Shape of referrals dataframe:", ref.shape)
    #Create a wide version of the target column
    ref['datetime']=pd.to_datetime(ref[c_r['date_col_out']], format= c_r['date_for_out'])
    ref_w=ref.pivot_table(index=c_r['per_col'], columns='datetime', values=c_e['ref_target'], aggfunc='sum')
    ref_w=ref_w.fillna(0) #fill in NA so sums correctly. 
    
    if pred.empty == True:
        pred_file=c_p['dir']+c_p['file']
        print("Loading predictions dataframe..", c_p['file'])
        pred=pd.read_csv(pred_file)
        print("Shape of referrals dataframe:", pred.shape)
    else:
        print("Shape of referrals dataframe:", pred.shape)

    #Reduce to the prediction being evaluated.
    pred['datetime']=pd.to_datetime(pred[c_p['date_col']], format= c_p['date_for'])
    
    pred=pred.loc[pred['datetime']==c_e['eval_date'],[c_p['per_col'],'datetime',c_e['pred_target']]]
    print("Shape of referrals dataframe:", pred.shape)
 
    #Initialize a results data frame.
    results=pd.DataFrame()
    row=0
    #Loop through different windows. 
    for w in c_e['landmarks']:
        start=c_e['eval_date']+pd.DateOffset(months=w[0])
        end=c_e['eval_date']+pd.DateOffset(months=w[1])
        label=start.strftime(c_r['date_for_out'])+'-'+ end.strftime(c_r['date_for_out'])
        print("Splitting dataset for evaluation at", c_e['eval_date'], "Evaluating from:", label)
        sl=slice(start,end)
        #take slice based on window
        ref_w=ref_w.fillna(0)
        y= ref_w.loc[:,sl]#take slice based on window
        print("Examining Columns Slice:", str(y.columns))
        y= y.sum(axis=1)#take slice based on window
        #If more than 1 referral in window, recode to 1
        y[y>1]=1
        #filter out people who arn't in the pred
        y=y[y.index.isin(pred[c_p['per_col']])]
        #print("after filtering out people not in prediction", pred.shape[0], len(y))
        
        #add 0s for people who aren't in ref.
        y=y.append(pd.Series(0,index=set(pred[c_p['per_col']])-set(y.index))).sort_index()
        #print("after adding in 0s", pred.shape[0], len(y))
        
        if pred.shape[0]!=len(y):
            print("ERROR: PREDICTION AND ACTUAL DATAFRAME HAVE DIFFERENT NUMBERS.  PREDICTION:",pred.shape[0], " EVALUATION: ",len(y))
            break
        results.loc[row, 'experiment']=c_e['experiment']
        results.loc[row, 'start_time']=pd.Timestamp.now(tz=None)
        results.loc[row, 'pred_dir']=c_p['dir']
        results.loc[row, 'pred_file']=c_p['file']
        results.loc[row, 'n'] = pred.shape[0]
        results.loc[row, 'range']=label
        results.loc[row, 'log_loss'] = log_loss(y, pred[c_e['pred_target']])
        results.loc[row, 'roc_auc_score'] = roc_auc_score(y, pred[c_e['pred_target']])

        for k in c_e['k']:
            col_label='_'+c_e['ref_target']+'_@k='+str(k)
            pred[col_label] = prob_to_bin(pred[c_e['pred_target']], k)
            results=add_results(results, row, y, pred[col_label], col_label)

        for p in c_e['thresholds']:
            col_label='_'+c_e['ref_target']+'_p>'+str(p)
            pred[col_label] = np.where(pred[c_e['pred_target']] > p, 1, 0)
            results=add_results(results, row, y, pred[col_label], col_label)
        results.loc[row, 'end_time']=pd.Timestamp.now(tz=None)
        results.loc[row, 'elapsed_time']=results.loc[row, 'end_time']- results.loc[row, 'start_time']
        row=row+1
    if c_e['save']:
        results_file=c_e['dir']+c_e['file']
        if c_e['append'] and os.path.exists(results_file):
            with open(results_file, 'a') as f:
                results.to_csv(f, header=False, index = False)
        else:
            results.to_csv(results_file, index = False)
    return results

def add_results(results, row, y, predictions, text):
    results.loc[row, 'precision'+text]=precision_score(y, predictions)
    results.loc[row, 'recall'+text]=recall_score(y, predictions)
    results.loc[row, 'accuracy'+text]=accuracy_score(y, predictions)
    results.loc[row, 'balanced_accuracy'+text]=balanced_accuracy_score(y, predictions)
    results.loc[row, 'f1'+text]=f1_score(y, predictions)
    return results

#merge needs to fill some NAs. 
def fill_na(df, patterns, value, c_type):
    for pattern in patterns:
        cols=df.columns[df.columns.str.contains(pattern)]
        for x in cols:
            df[x]=df[x].fillna(value).astype(c_type)
    return df

def generate_test_prediction_files(c_p, c_r, patients, startdate, enddate):
    """
    This takes data and backs into 
    """
    #ref (dataframe) = "/data/referrals/test.csv" - ['person_id', 'date', 'class'] - 23 referrals
    # -> NO NEED: ref=pd.read_csv(c_r['dir']+c_r['file'])

    ref, trans = preprocess_referrals(c_r)
    cols=['ref','lab_1','lab_2','lab_3']
    pcols=['pref','plab_1','plab_2','plab_3']
    #Create a starter matrix with all 0
    s1 = pd.Series(range(0,patients))
    s2 = pd.date_range(startdate,enddate, freq='MS').strftime("%Y%m").astype(int)
    pred = pd.DataFrame(list(itertools.product(s1,s2)),columns=[c_p['per_col'],c_p['date_col']])
    for col in pcols:
        pred[col]=0
    
    df_dates_ref=pd.DataFrame()
    ref_temp=ref.copy()
    df_dates_ref['datetime']=pd.to_datetime(ref[c_r['date_col_out']], format= c_r['date_for_out'])
    #Loop through -12 months to +12 by 6 month. 
    for x in range(-12,13,6):
        ref_temp=ref.copy()
        sh="shift"+str(x)
        print(sh)
        df_dates_ref[sh]= df_dates_ref['datetime']+ pd.DateOffset(months=x)
        df_dates_ref[sh]=df_dates_ref[sh].dt.strftime(c_r['date_for_out']).astype(int)
        ref_temp[c_r['date_col_out']]=df_dates_ref[sh]
        df=pd.merge(pred, ref_temp, how='left',  on=[c_r['per_col'], c_r['date_col_out']])
        df=fill_na(df,['lab_','ref'],0, int)
        df[pcols]=df[cols]
        df.drop(columns=cols, inplace=True, axis=0)
        print("Saving dataframe.  Records:", df.shape[0], "Patients",patients)
        df.to_csv('../data/predictions/tests/tests_100_'+sh+'.csv', index=False)
        df[df==1]=0.75
        df.to_csv('../data/predictions/tests/tests_75_'+sh+'.csv', index=False)
        
def prob_to_bin(target, k):
    ind=np.argpartition(target, -k)[-k:]
    target_bin=np.zeros(len(target))
    target_bin[ind]=1
    return target_bin
