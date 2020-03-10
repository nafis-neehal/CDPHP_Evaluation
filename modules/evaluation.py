#Set config.
import sys, datetime, os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, accuracy_score, recall_score, balanced_accuracy_score, f1_score, roc_auc_score, log_loss, roc_curve

def preprocess_referrals(c_r, drop_duplicates=True ):
    """
    Preprocess referral data, aggregating reason codes, and referrals.
    """
    ref=pd.read_csv(c_r['dir']+c_r['file'])
    #Translate Dates to the datetime format.
    ref['datetime']=pd.to_datetime(ref[c_r['date_col_in']], format= c_r['date_for_in'])
    ref[c_r['date_col_out']]=ref['datetime'].dt.strftime(c_r['date_for_out']).astype(int)

    #This creates a class number for each reason code
    trans=ref[c_r['reason_col']].unique()
    translate=dict(zip(ref[c_r['reason_col']].unique(),[x for x in range(1,len(trans)+1)]))

    #Select only the important columns.
    ref=ref.loc[:,[c_r['per_col'],c_r['date_col_out'], c_r['reason_col'],'datetime']]

    #This creates a column with a label consistent with translation.
    ref['label']=ref[c_r['reason_col']].map(lambda x: translate[x])
    lab_dum=pd.get_dummies(ref['label'],prefix='lab')
    ref = pd.concat([ref, lab_dum], axis=1);
    #This sums up the referrals.
    ref = pd.pivot_table(ref, values=lab_dum.columns, index=[c_r['per_col'],c_r['date_col_out']], aggfunc=np.sum)
    #This sums the total number of referrals.
    #ref['ref_m']=ref.sum(axis=1)

    #This creates a binary variable for referial.
    ref['ref']=1

    if drop_duplicates==True:
        cols=list(ref.columns[ref.columns.str[0:len('lab_')]=='lab_'])
        for c in cols:
            ref[c]=ref[c].map(lambda z: 1 if z>=1 else 0)
    #reset the index
    ref.reset_index(inplace=True)
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

    if pred.empty == True:
        pred_file=c_p['dir']+c_p['file']
        print("Loading predictions dataframe..", c_p['file'])
        pred=pd.read_csv(pred_file)
    else:
        print("Shape of referrals dataframe:", pred.shape)

    #Reduce to the prediction being evaluated.
    pred['datetime']=pd.to_datetime(pred[c_p['date_col']], format= c_p['date_for'])

    pred=pred.loc[pred['datetime']==c_e['eval_date'],[c_p['per_col'],'datetime',c_e['pred_target']]]

    #Initialize a results data frame.
    results=pd.DataFrame()
    row=0

    for w in c_e['landmarks']:
        start=c_e['eval_date']+pd.DateOffset(months=w[0])
        end=c_e['eval_date']+pd.DateOffset(months=w[1])
        label=start.strftime(c_r['date_for_out'])+'-'+ end.strftime(c_r['date_for_out'])
        print("Evaluating from:", label)
        sl=slice(start,end)
        #take slice based on window
        y= ref_w.loc[:,sl].sum(axis=1)#take slice based on window
        #If more than 1 referral in window, recode to 1
        y[y>1]=1
        #filter out people who arn't in the pred
        y=y[y.index.isin(pred[c_p['per_col']])]
        #add 0s for people who aren't in ref.
        y=y.append(pd.Series(0,index=set(pred[c_p['per_col']])-set(y.index))).sort_index()

        #if pred.shape[0]!=len(y):
        #    print("df with ",y, " people;",pred.shape[0], " predictions" )
        #    exit

        results.loc[row, 'experiment']=c_e['experiment']
        results.loc[row, 'date']=pd.Timestamp.now(tz=None)
        results.loc[row, 'pred_dir']=c_p['dir']
        results.loc[row, 'pred_file']=c_p['file']
        results.loc[row, 'n'] = pred.shape[0]
        results.loc[row, 'range']=label
        results.loc[row, 'log_loss'] = log_loss(y, pred[c_e['pred_target']])
        results.loc[row, 'roc_auc_score'] = roc_auc_score(y, pred[c_e['pred_target']])
          #loop through to evaluate for different K
        # #for lim in c_e['k']:
        # #    results.loc[row, 'precision@'+str(lim)]=precision_score(y, pred[c_e['pred_target']])
        #     results.loc[row, 'recall@'+str(lim)]=recall_score(y, pred[c_e['pred_target']])
        #     results.loc[row, 'accuracy@'+str(lim)]=accuracy_score(y, pred[c_e['pred_target']])
        #     results.loc[row, 'balanced_accuracy@'+str(lim)]=balanced_accuracy_score(y, pred[c_e['pred_target']])
        #     results.loc[row, 'f1@'+str(lim)]=f1_score(y, pred[c_e['pred_target']])
        row=row+1
    if c_e['save']:
        results_file=c_e['dir']+c_e['file']
        if c_e['append'] and os.path.exists(results_file):
            with open(results_file, 'a') as f:
                results.to_csv(f, header=False)
        else:
            results.to_csv(results_file, index = False)
