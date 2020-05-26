import Helper
import Score
import pandas as pd
import numpy as np

#cp, c_r, c_e are all mutable
#mutable obj integrity checked
def evaluate(c_p, c_r, c_e):
    
    eval_method = c_e['eval_method']      
    
    referral = pd.read_csv(c_r['dir'] + c_r['file'])
    ref_copy = referral.copy() #shallow copy
    
    #convert_dates_ref(c_r)
    if c_r['date_format']=='ym':
        Helper.ym_to_datetime(c_r, ref_copy) #passing reference of mutable ref_copy, will be directly edited
    else:
        ref_copy[c_r['columns']['date_column']] = pd.to_datetime(ref_copy[c_r['columns']['date_column']])
        Helper.convert_dates_ref(ref_copy, c_r)
    
    #now both pred_copy['Date'] and ref_copy['Date'] is in Datetime format instead of previous String format
    
    prediction = pd.read_csv(c_p['dir'] + c_p['file'])
    pred_copy = prediction.copy()
        
    #change date from ym to ymd here
    if c_p['date_format']=='ym':
        Helper.ym_to_datetime(c_p, pred_copy) #passing reference of mutable pred_copy, will be directly edited
    else:
        pred_copy[c_p['columns']['date_column']] = pd.to_datetime(pred_copy[c_p['columns']['date_column']])
        
    #now both ref_copy and pred_copy are datetime
    ##only applying top_k in selected range, not the whole column-------------------->>>>>>>
    pred_copy = pred_copy.loc[pred_copy[c_p['columns']['date_column']]==c_e['eval_date']] 
    
    all_model_list = c_e['eval_models']
    
    #if all models predicts -1, then I can safely drop patient-month-date row with -1 predictions here
    #select rows, where the first model is not -1 (all model predicts -1)
    if c_e['drop_neg_prob'] == True:
        pred_copy = pred_copy[pred_copy[all_model_list[0]]!=-1]
    
    #remove recent referrals checking back k months in referral and drop them from prediction
    if c_e['drop_ref'] == True:
        pred_copy = Helper.drop_recent_referrals(c_p, c_r, c_e, ref_copy, pred_copy)
    
    #pivot referral table
    ref_copy['target'] = pd.Series(np.ones(ref_copy.shape[0], dtype=float))
    ref_copy = ref_copy.pivot_table(index=c_r['columns']['id_column'], columns=c_r['columns']['date_column'], values='target', aggfunc='sum')
    ref_copy = ref_copy.fillna(0)
    
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
            y_true = ref_copy.loc[:,sl] #also a shallow copy
            
            #aggregate referrals
            y_true = y_true.sum(axis=1)
            
            #aggregated referral map to 1 if >1
            y_true[y_true>1] = 1
            
            #for the time being, all ref patients are in pred_list. 
            #Change random.randint upper_range in data_generate to tweak
            y_true=y_true[y_true.index.isin(pred_copy[c_p['columns']['id_column']])]
            
            y_true=y_true.append(pd.Series(0,index=set(pred_copy[c_p['columns']['id_column']])-set(y_true.index))).sort_index()
        
            #their size have to be same -- TO USE the built in metric functions
            if pred_copy.shape[0]!=len(y_true):
                print("ERROR: PREDICTION AND ACTUAL DATAFRAME HAVE DIFFERENT NUMBERS.  PREDICTION:",
                      pred_copy.shape[0], " EVALUATION: ",len(y_true))
                break
            
            #now thresholding method
            if eval_method=='top_k':
                for k_values in c_e['top_k']:
                    label = model+'_window_['+str(window[0])+','+str(window[1])+']_'+eval_method + '_@' + str(k_values)
                    pred_copy[label] = Helper.prob_to_bin(pred_copy[model], k_values)
                    #model score for this (window,k) update
                    update_model_score(model, evaluated_model_obj, label, y_true, pred_copy)
    
            elif eval_method == 'thresholding':
                for thresholds in c_e['thresholding']:
                    label = model+'_window_['+str(window[0])+','+str(window[1])+']_'+eval_method + '_@' + str(thresholds)
                    pred_copy[label] = np.where(pred_copy[model] > thresholds, 1, 0)
                    #model score for this (window,threshold) update
                    update_model_score(model, evaluated_model_obj, label, y_true, pred_copy)
            
        all_model_evaluations.update({model:evaluated_model_obj})
    
    return all_model_evaluations

#mutable obj integrity checked
#evaluated_model_obj changed as requirement
def update_model_score(model, evaluated_model_obj, label, y_true, pred_copy):
    
    precision = evaluated_model_obj.get_precision(y_true.values, pred_copy[label].values)
    evaluated_model_obj.precision.update({label:precision})

    recall = evaluated_model_obj.get_recall(y_true.values, pred_copy[label].values)
    evaluated_model_obj.recall.update({label:recall})

    accuracy = evaluated_model_obj.get_accuracy(y_true.values, pred_copy[label].values)
    evaluated_model_obj.accuracy.update({label:accuracy})

    balanced_acc = evaluated_model_obj.get_balanced_acc(y_true.values, pred_copy[label].values)
    evaluated_model_obj.balanced_acc.update({label:balanced_acc})

    f1_score = evaluated_model_obj.get_f1_score(y_true.values, pred_copy[label].values)
    evaluated_model_obj.f1_score.update({label:f1_score})
    
    confusion_matrix = evaluated_model_obj.get_confusion_matrix(y_true.values, pred_copy[label].values)
    evaluated_model_obj.confusion_matrix.update({label:confusion_matrix})

    #these will take predicted probabilites, not thresholded binaries

    log_loss = evaluated_model_obj.get_log_loss(y_true.values, pred_copy[model].values)
    evaluated_model_obj.log_loss.update({label:log_loss})

    roc_auc_score = evaluated_model_obj.get_roc_auc_score(y_true.values, pred_copy[model].values)
    evaluated_model_obj.roc_auc_score.update({label:roc_auc_score})
    
    brier_score_loss = evaluated_model_obj.get_brier_score_loss(y_true.values, pred_copy[model].values)
    evaluated_model_obj.brier_score_loss.update({label:brier_score_loss})
    
    #add number of samples used by this model
    evaluated_model_obj.experimental_samples = y_true.shape[0]
    
    