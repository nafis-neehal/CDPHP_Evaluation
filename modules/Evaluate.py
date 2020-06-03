import Helper
import Score
import Aws
import pandas as pd
import numpy as np
from IPython.core.display import display

#cp, c_r, c_e are all mutable
#mutable obj integrity checked
def evaluate(c_p, c_e, c_r, referral, prediction ):

    #assert if referral and prediction dfs both are non-empty
    assert (not(referral.empty)), "Referral is empty at the beginning of evaluation"
    assert (not(prediction.empty)), "Prediction is empty at the beginning of evaluation"
    
    eval_method = c_e['eval_method']      
       
    #convert_dates_ref(c_r)
    if c_r['date_format']=='ym':
        df = Helper.ym_to_datetime(c_r, referral) #passing reference of mutable referral, will be directly edited
    else:
        referral[c_r['columns']['date_column']] = pd.to_datetime(referral[c_r['columns']['date_column']])
        Helper.convert_dates_ref(referral, c_r)

     
    #change date from ym to ymd here
    if c_p['date_format']=='ym':
        Helper.ym_to_datetime(c_p, prediction) #passing reference of mutable prediction, will be directly edited
    else:
        prediction[c_p['columns']['date_column']] = pd.to_datetime(prediction[c_p['columns']['date_column']])

    #check if eval date exists in prediction file 
    assert (sum(prediction[c_p['columns']['date_column']]==c_e['eval_date'])!=0), "Evaluation Date " + c_e['eval_date'] + " doesn't exist in prediction"
        
    #now both referral and prediction are datetime
    ##only applying top_k in selected range, not the whole column-------------------->>>>>>>
    prediction = prediction.loc[prediction[c_p['columns']['date_column']]==c_e['eval_date']] 
    all_model_list = c_p['eval_models']
    
    #if all models predicts -1, then I can safely drop patient-month-date row with -1 predictions here
    #select rows, where the first model is not -1 (all model predicts -1)
    if c_e['drop_neg_prob'] == True:
        prediction = prediction[prediction[all_model_list[0]]!=-1]
        assert (not(prediction.empty)), "Prediction df is empty after dropping negative probability"

    #remove recent referrals checking back k months in referral and drop them from prediction
    if c_e['drop_ref'] == True:
        prediction = Helper.drop_recent_referrals(c_p, c_r, c_e, referral, prediction)
        assert (not(prediction.empty)), "Prediction df is empty after dropping recent referrals"

    #pivot referral table
    referral['target'] = pd.Series(np.ones(referral.shape[0], dtype=float))
    referral = referral.pivot_table(index=c_r['columns']['id_column'], columns=c_r['columns']['date_column'], values='target', aggfunc='sum')
    referral = referral.fillna(0)
    
    all_model_evaluations = {} #{'model_name':score class object for that model}
   
    #now branch out for each model
    for model in all_model_list:
        
        evaluated_model_obj = Score.Score(model)
        
        #calculate for each window
        for window in c_e['eval_windows']:
            
            #taking negative window into consideration
            if(window[0]<0):
                start = c_e['eval_date'] - pd.DateOffset(months=(-1)*window[0])
            else:
                start = c_e['eval_date'] + pd.DateOffset(months=window[0])
                
            end = c_e['eval_date'] + pd.DateOffset(months=window[1])
            
            sl=slice(start,end)
            y_true = referral.loc[:,sl] #also a shallow copy
            assert (not(y_true.empty)), "Referral y_true df is empty after window slicing"
            
            #aggregate referrals
            y_true = y_true.sum(axis=1)
            
            #aggregated referral map to 1 if >1
            y_true[y_true>1] = 1
            
            #for the time being, all ref patients are in pred_list. 
            #Change random.randint upper_range in data_generate to tweak
            y_true=y_true[y_true.index.isin(prediction[c_p['columns']['id_column']])]
            assert (not(y_true.empty)), "Referral y_true df is empty after overlapping with prediction df"
            
            y_true=y_true.append(pd.Series(0,index=set(prediction[c_p['columns']['id_column']])-set(y_true.index))).sort_index()
        
            #their size have to be same -- TO USE the built in metric functions
            if prediction.shape[0]!=len(y_true):
                print("ERROR: PREDICTION AND ACTUAL DATAFRAME HAVE DIFFERENT NUMBERS.  PREDICTION:",
                      prediction.shape[0], " EVALUATION: ",len(y_true))
                break
            
            #now thresholding method
            if eval_method=='top_k':
                for k_values in c_e['top_k']:
                    label = model+'_window_['+str(window[0])+','+str(window[1])+']_'+eval_method + '_@k=' + str(k_values)
                    prediction[label] = Helper.prob_to_bin(prediction[model], k_values)
                    #model score for this (window,k) update
                    update_model_score(model, evaluated_model_obj, label, y_true, prediction)
    
            elif eval_method == 'thresholding':
                for thresholds in c_e['thresholding']:
                    label = model+'_window_['+str(window[0])+','+str(window[1])+']_'+eval_method + '_@p>=' + str(thresholds)
                    prediction[label] = np.where(prediction[model] > thresholds, 1, 0)
                    #model score for this (window,threshold) update
                    update_model_score(model, evaluated_model_obj, label, y_true, prediction)

            elif eval_method == 'both':
                for k_values in c_e['top_k']:
                    label = model+'_window_['+str(window[0])+','+str(window[1])+']_'+ 'top_k' + '_@k=' + str(k_values)
                    prediction[label] = Helper.prob_to_bin(prediction[model], k_values)
                    #model score for this (window,k) update
                    update_model_score(model, evaluated_model_obj, label, y_true, prediction)
                for thresholds in c_e['thresholding']:
                    label = model+'_window_['+str(window[0])+','+str(window[1])+']_'+ 'thresholding' + '_@p>=' + str(thresholds)
                    prediction[label] = np.where(prediction[model] > thresholds, 1, 0)
                    #model score for this (window,threshold) update
                    update_model_score(model, evaluated_model_obj, label, y_true, prediction)
            
        all_model_evaluations.update({model:evaluated_model_obj})
    
    return all_model_evaluations

#mutable obj integrity checked
#evaluated_model_obj changed as requirement
def update_model_score(model, evaluated_model_obj, label, y_true, prediction):
    
    precision = evaluated_model_obj.get_precision(y_true.values, prediction[label].values)
    evaluated_model_obj.precision.update({label:precision})

    recall = evaluated_model_obj.get_recall(y_true.values, prediction[label].values)
    evaluated_model_obj.recall.update({label:recall})

    accuracy = evaluated_model_obj.get_accuracy(y_true.values, prediction[label].values)
    evaluated_model_obj.accuracy.update({label:accuracy})

    balanced_acc = evaluated_model_obj.get_balanced_acc(y_true.values, prediction[label].values)
    evaluated_model_obj.balanced_acc.update({label:balanced_acc})

    f1_score = evaluated_model_obj.get_f1_score(y_true.values, prediction[label].values)
    evaluated_model_obj.f1_score.update({label:f1_score})
    
    confusion_matrix = evaluated_model_obj.get_confusion_matrix(y_true.values, prediction[label].values)
    evaluated_model_obj.confusion_matrix.update({label:confusion_matrix})

    #these will take predicted probabilites, not thresholded binaries

    log_loss = evaluated_model_obj.get_log_loss(y_true.values, prediction[model].values)
    evaluated_model_obj.log_loss.update({label:log_loss})

    roc_auc_score = evaluated_model_obj.get_roc_auc_score(y_true.values, prediction[model].values)
    evaluated_model_obj.roc_auc_score.update({label:roc_auc_score})
    
    brier_score_loss = evaluated_model_obj.get_brier_score_loss(y_true.values, prediction[model].values)
    evaluated_model_obj.brier_score_loss.update({label:brier_score_loss})
    
    #add number of samples used by this model
    evaluated_model_obj.experimental_samples = y_true.shape[0]
    
    #add number of samples used by this model
    evaluated_model_obj.experimental_samples = y_true.shape[0]
    
    
