
This fixed the issue with missmatched. 
```
prediction.drop_duplicates(subset=['PERSON_ID','MYR'], inplace=True)
```



Changes to be implemented: 

THIS IS CAUSING ISSUES: 
 prediction = prediction.loc[prediction[c_p['columns']['date_column']]==c_e['eval_date']] 




```
import numpy as np
r=set("a total of referrals['person_id'].unique())
p=set(prediction0['PERSON_ID'].unique())
print(len(r),len(p))
result=r.intersection(p)
len(result)

```

```import pandas as pd
import numpy  as np


def data_evaluation(referal_matrix = None, probabilities = None, lookback = 2, forward_window = 6,
                   backward_window = 3, threshold = 0.95, k = -1):
    
    """
    A FUNCTION FOR A SIMULATED EVALUATION BASED ON TIME SERIES
    
    referal_matrix: 2D pandas matrix with indexes the patients IDs and Columns the Months
                     it has 1 at the date of referal and 0 else
                     
    probabilities: 2D pandas matrix with columsn: PERSON_ID, MYR, model_probas(the name of this can
    be arbitrary, the first two names should be the same PERSON_ID, MYR)
                    the probabilities for a specific model
                    
    lookback     : integer, how many consecutive 1s the model needs to predict in order
                   to be considered 1: ex: lookback = 2 
                                           predictions: 0 0 0 1 1 1 1
                             predictions with lookback: 0 0 0 0 0 1 1
                             
    forward_window: How far in the future our prediction holds true
                                        ex: forward_window = 3
                                        prediction at month i: 1
                                        Ground truth at months i, i+1, i+2, i+3: 0 0 0 1
                                        conclusion: Prediction was a True positive
                                                    because the patient was refered
                                                    inside the prediction horizon
                                                    
    backward_window: How far in the past our prediction holds true:
    
                                        ex: backward_window = 2
                                        prediction at month i: 1
                                        Ground Truth at months i-2, i-1, i, i+1: 1 0 0 0
                                        conclusion: Prediction was a True Positive because
                                        even though the labels at times i-1, i, i+1 were
                                        0 the label at time i-2 was 1 so we predicted correct
                                        into the past horizon of prediction.
                                        
    threshold: float: where to threshold the probabilities
    
    k = -1 use threshold else top k points as 1s
    
    returns: Pandas Data Frame With Metrics Per Month
                    
    """
   
    
    ref_test_sm = referal_matrix.copy()
    
    
    #get the distinct months
    months = sorted(list(ref_test_sm.columns.astype(int)))
    probabilities = probabilities.copy()
    probabilities.PERSON_ID = probabilities.PERSON_ID.astype(int)
    p2 = probabilities.iloc[:,0:2]
    
    #making the matrix PERSON_ID, MYR probas
    if  k == -1:
        p2['probas'] = (probabilities.iloc[:,2] >= threshold).astype(int)
    else:
        
        p2['probas'] = 0
        for month in months:
            ii = p2[p2.MYR == month].iloc[:,2].sort_values()[-k:].index
            p2.probas.iloc[ii] = 1
            
        
        
    p3  = p2.copy()
    p3['todrop'] = (p3.PERSON_ID.astype(int)*p3.MYR)
    
    #remove duplicates
    p4 = p3.drop_duplicates(subset = 'todrop')
    p4.reset_index(drop = True, inplace = True)
    p4 = p4.drop(columns = 'todrop')
    
    #MAKE A MULTI INDEX DATAFRAME
    p5 = p4.set_index(['PERSON_ID', 'MYR'])

    #MANIPULATE THIS DATAFRAME FOR OUR PURPOSE OF LOOKBACK
    p6 = p5.unstack(fill_value = -1)#.stack().reset_index()
    #test6 = test6.drop(columns = 'todrop')
    
    #changing the type of the ref index
    ref_test_sm.index = ref_test_sm.index.astype(int)
    
    K = forward_window
    K_b = backward_window
    lk = lookback
    h  = max(K_b, lk)
    mets = []
    for i in range(h, len(months)-K):
        month_test = months[i-K_b:i+1+K]
        month_predict = months[i]
    
        #get referrals for that month
        reft = ref_test_sm[month_test]
    
        #get ids for predict month
        mask =  p4.MYR == month_predict
        ids  =  p4[mask].PERSON_ID
    
        #prob =  test3[mask].probas.values
        prob  = np.min(p6.loc[ids].iloc[:, i-lk: i+1].values, axis = 1)
    
        #new  = test3[mask]
        reft2 = np.max(reft.loc[ids].values, axis = 1)
        TP =    np.sum((prob == 1) & (reft2 == 1))
        FP =    np.sum((prob == 1) & (reft2 == 0))
        TN =    np.sum((prob == 0) & (reft2 == 0))
        FN =    np.sum((prob == 0) & (reft2 == 1))
        prec = TP/(TP+FP)
        acc  = (TP+TN)/(TP+TN+FP+FN)
        mets.append([TP, FP, TN, FN, prec, acc])
        
    mets_pd = pd.DataFrame(np.array(mets), index = months[h:-K], 
                           columns = ['TP','FP', 'TN', 'FN', 'prec','acc'])
    
    return mets_pd#, p6



def lk_probas(probas = None, lk = 0, drop_old = False):
    
    """
    probas: pandas Array in the format
    PERSON_ID MYR <probabilities>
    
    the probabilities column can have any name
    
    lk : the look back to compute the probabilities
    drop_old: If the returned array will have both
              lookback probas or not
    
    """
    #get different months
    probas = probas.copy()
    months = list(sorted(probas.MYR.unique()))
    
    #copy probas drp duplicates
    probas1 = probas.copy()
    probas1['to_drop'] = probas1.MYR.values * probas1.PERSON_ID.values
    probas1 = probas1.drop_duplicates(subset = 'to_drop')
    probas1 = probas1.drop(['to_drop'], axis = 1)
    
    #get 2d probas
    probas_hier = probas1.copy().set_index(['PERSON_ID', 'MYR'])
    probas_2d = probas_hier.copy().unstack(fill_value = -1)
    
    #probas lk
    probas_lk = probas1.copy()
    c_name = probas_lk.columns[2]
    c_name = c_name+'_lk'+str(lk)
    probas_lk[c_name] = -1
    
    #take lookback probs
    for i, mon in enumerate(months):
        mask = probas1.MYR == mon
        ids = probas1[mask].PERSON_ID
        
        #print(mon)
        if i -lk < 0:
            continue
        else:
           # print('here')
            probs_help = np.min(probas_2d.loc[ids].iloc[:, i-lk: i+1].values, axis = 1)
            probas_lk.loc[mask, c_name] = probs_help
    
    if drop_old:
        probas_lk = probas_lk.iloc[:,[0,1,3]]
        
    return probas_lk

def lk_probas_multiple(probas = None, lk = 0, drop_old = False):
    
    """
    Same as lk_probas
    but now probas can have multiple probabilities
    PERSON_ID MYR p1 p2 p3 .......
    
    """
    probas = probas.copy()
    #iterate for all probasbility columns
    for i in range(probas.shape[1]-2):
        
        out = lk_probas(probas = probas.copy().iloc[:,[0,1, i+2]], lk = lk, drop_old = True)
        
        if i == 0:
            
            proba_lk = out.copy()
            
        else:
            #get the name of the columns
            
            column_name = out.columns[2]
            proba_lk[column_name] = out.iloc[:,2]
            
    
    return proba_lk
        ```
