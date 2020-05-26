import Helper
import pandas as pd
import numpy as np
import itertools as it

#for generaiton
#mutable obj integrity checked
#no changes in c_p, c_r
def generate_synthetic_ground_truth_data(c_gen, c_r):
    seed = 1234
    np.random.seed(seed)
    patients = pd.Series([x for x in np.random.randint(0, c_gen['ref']['upper_bound'], c_gen['ref']['num_samples'])])
    patients = patients.sort_values(ascending=True).reset_index(drop=True)
    date = pd.Series([d for d in Helper.generate_random_date(c_gen['ref']['start_date'], c_gen['ref']['end_date'], c_gen['ref']['num_samples'], c_r['date_format'])])
    data = pd.DataFrame({c_r['columns']['id_column']:patients, c_r['columns']['date_column']:date})
    data.to_csv(c_r['dir']+c_r['file'], index = False, float_format= '%8.5f', mode='w')

#for generation
#mutable obj integrity checked
#no changes in c_p
def generate_synthetic_prediction_data(c_gen, c_p):
    patients = pd.Series(range(0, c_gen['pred']['num_samples'])) #500
    
    if c_p['date_format']=='ym':
        date = pd.date_range(c_gen['pred']['start_date'], c_gen['pred']['end_date'], freq='MS').strftime("%Y%m").astype(int) #25
    elif c_p['date_format']=='y-m-d':
        date = pd.date_range(c_gen['pred']['start_date'], c_gen['pred']['end_date'], freq='MS').strftime("%Y-%m-%d")
    else:
        date = pd.date_range(c_gen['pred']['start_date'], c_gen['pred']['end_date'], freq='MS')
    
    data = pd.DataFrame(list(it.product(patients,date)),columns=[c_p['columns']['id_column'],c_p['columns']['date_column']])
    seed = 1234
    for model in c_gen['pred']['model_columns']:
        np.random.seed(seed)
        data[model] = pd.Series(np.random.random((data.shape[0])))
        seed+=1
    
    #dataframe_to_csv(data, c_p['dir']+c_p['file'])
    data.to_csv(c_p['dir']+c_p['file'], index=False, float_format= '%8.5f', mode='w')