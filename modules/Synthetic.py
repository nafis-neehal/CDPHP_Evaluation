import Helper
import Aws
import pandas as pd
import numpy as np
import itertools as it

#for generaiton
#mutable obj integrity checked
#no changes in c_p, c_r
def generate_synthetic_event_data(c_gen ):
    seed = 1234
    np.random.seed(seed)
    persons = pd.Series([x for x in np.random.randint(0, c_gen['ref']['upper_bound'], c_gen['ref']['num_samples'])])
    persons = persons.sort_values(ascending=True).reset_index(drop=True)
    date = pd.Series([d for d in Helper.generate_random_date(c_gen['ref']['start_date'], c_gen['ref']['end_date'], c_gen['ref']['num_samples'], c_gen['ref']['date_format'])])
    data = pd.DataFrame({c_gen['ref']['columns']['id_column']:persons, c_gen['ref']['columns']['date_column']:date})
    data.to_csv(c_gen['dir']+c_gen['ref']['file'], index = False, float_format= '%8.5f', mode='w')
    #if c_e['aws'] == True:
     #    Aws.upload_to_aws(c_r['dir']+c_r['file'], c_r['bucket'], c_r['dir']+c_r['file'])

#for generation
#mutable obj integrity checked
#no changes in c_p
def generate_synthetic_prediction_data(c_gen ):
    patients = pd.Series(range(0, c_gen['pred']['num_samples'])) #500
    
    if c_gen['pred']['date_format']=='ym':
        date = pd.date_range(c_gen['pred']['start_date'], c_gen['pred']['end_date'], freq='MS').strftime("%Y%m") #25
    elif c_gen['pred']['date_format']=='y-m-d':
        date = pd.date_range(c_gen['pred']['start_date'], c_gen['pred']['end_date'], freq='MS').strftime("%Y-%m-%d")
    else:
        date = pd.date_range(c_gen['pred']['start_date'], c_gen['pred']['end_date'], freq='MS')
        
        
    data = pd.DataFrame(list(it.product(patients,date)),columns=[c_gen['pred']['columns']['id_column'],c_gen['pred']['columns']['date_column']])
    seed = 1234
    for model in c_gen['pred']['model_columns']:
        np.random.seed(seed)
        data[model] = pd.Series(np.random.random((data.shape[0])))
        seed+=1
    
   
    data.to_csv(c_gen['dir']+c_gen['pred']['file'], index=False, float_format= '%8.5f', mode='w')
    #if c_e['aws'] == True:
    #            Aws.upload_to_aws(c_p['dir']+c_p['file'], c_p['bucket'], c_p['dir']+c_p['file'])