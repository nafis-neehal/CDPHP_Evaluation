import pandas as pd
from IPython.core.display import display
import os 
import seaborn as sns
import matplotlib.pyplot as plt
import Helper, Aws 


#generate plottable dataframe from whole result: evaluation_to_dataframe() helper function
#visualize result (if visual==True): visualize_performance() helper_function
#save result to CSV file: Use dataframe_to_csv() helper function
#mutable obj integrity checked
def present_evaluation(c_p, c_r, c_e, c_visual, all_model_evaluations, table=False, plot = False, save=False):
    
    #result file
    result_file = c_e['dir'] + c_e['file']
    
    #clear if there is already any result file previously (from previous run/experiment), 
    #otherwise it will just keep appending, as opening in append mode
        
    data = generate_tabular_data(c_p, c_r, c_e, c_visual, all_model_evaluations)
    
    if c_visual['table']['show'] == True:
        pd.set_option('display.max_columns', 9999)
        pd.options.display.float_format = '{:8.5f}'.format
        display(data)
        
    #will add graph here
    if c_visual['model_comparison']['plot'] == True:
        generate_comparison_plot(c_e, data, c_visual['model_comparison']['windows'], c_visual['model_comparison']['metrics'])
    
    if c_visual['confusion_matrix']['plot'] == True:
        generate_confusion_matrix_plot(c_e, all_model_evaluations, c_visual['confusion_matrix']['model'], c_visual['confusion_matrix']['window'], 
                                       c_visual['confusion_matrix']['thres'])
    
    if c_visual['probability_distribution']['plot'] == True:
        generate_probability_distribution_plot(c_p, c_visual['probability_distribution']['models'])
    
    if c_e['save_csv'] == True:
            if os.path.exists(result_file):
                data.to_csv(result_file, index = False, float_format= '%8.5f', mode='a')
            else:
                data.to_csv(result_file, index = False, float_format= '%8.5f', mode='w')
            if c_e['aws'] == True:
                Aws.upload_to_aws(result_file, c_e['bucket'], result_file)
    
#name of models, windows, top_ks
def generate_confusion_matrix_plot(c_e, all_model_evaluations, model, window, thres):
    confusion_matrix = all_model_evaluations[model].confusion_matrix
    win = '[' + str(window[0]) + ',' + str(window[1]) + ']'
    th = str(thres)
    for keys in confusion_matrix.keys():
        if win in keys and th in keys:
            conf_mat = confusion_matrix[keys]
            conf_mat = pd.DataFrame(conf_mat)
            akws = {"ha": 'left',"va": 'top'}
            plt.figure()
            ax = sns.heatmap(conf_mat, annot=True, fmt="d", annot_kws=akws, cmap='Greens_r')
            ax.set_title("Confusion Matrix of " + model + ' for window ' + win + ' for ' + c_e['eval_method'] + '_@' + str(thres))

#input: c_p, list_of_models
#output: probability distribution plots
def generate_probability_distribution_plot(c_p, models):
    file = c_p['dir'] + c_p['file']
    pred = pd.read_csv(file)
    for each_model in models:
        plt.figure()
        ax = sns.distplot(pred[each_model])
        ax.set_title("Probability Distribution of " + each_model)

#generate comparison plot
def generate_comparison_plot(c_e, data, plot_window, metrics):
    
    for window in plot_window:
        
        win = Helper.window_to_range(c_e, window)
        new_data = data[data['Window']==win]

        for t in c_e[c_e['eval_method']]:
            metric_col = [col for col in new_data.columns if str(t) in col] #original column names for metrics (All 8 metric columns)
            final_original_columns = ['Model']
            final_modified_columns = ['Model']

            for metric_name in metrics:
                final_original_columns = final_original_columns + [col for col in metric_col if metric_name in col] #only columns passed
                final_modified_columns = final_modified_columns + [metric_name for col in metric_col if metric_name in col]

            pf = new_data[final_original_columns].copy()

            for i in range(len(pf.columns)):
                pf.rename(columns={final_original_columns[i]:final_modified_columns[i]}, inplace=True)

            #pivot the data to covert wide dataframe to long dataframe
            pf_new = pd.melt(pf, id_vars=['Model'], value_vars=final_modified_columns[1:])

            data_plot = sns.catplot(x="variable", y="value", hue="Model", data=pf_new, kind="bar", height = 4, aspect = 3)
            data_plot.set_xlabels("Metrics")
            data_plot.set_ylabels("Score")
            title = "Performance of Models During " + win + " with " + c_e['eval_method'] + "_@" + str(t)
            data_plot.fig.suptitle(title)

#takes all info, returns dataframe 
def generate_tabular_data(c_p, c_r, c_e, c_visual, all_model_evaluations):
    
    data = pd.DataFrame()
    row = 0
    metric_name = c_visual['table']['metrics']
    #conf_mat = {"TN":0, "FP":0, "FN":0, "TP":0}
    conf_head = ["TN", "FP", "FN", "TP"]
    
    for model in all_model_evaluations:
        for window in c_e['eval_windows']:
            sub_key = '['+str(window[0])+','+str(window[1])+']'
            data.loc[row, 'Start Time'] = pd.Timestamp.now(tz='US/Eastern')
            data.loc[row, 'Experiment Name'] = c_e['experiment']
            data.loc[row, 'Model'] = model
            data.loc[row, 'Window'] = Helper.window_to_range(c_e, window)
            data.loc[row, 'Eval Date'] = c_e['eval_date']
            data.loc[row, 'Num Samples'] = all_model_evaluations[model].experimental_samples
            
            #-- change here if both k and thresh needed
            for metric in metric_name:
                
                values  = [value for key, value in all_model_evaluations[model].__dict__[metric].items() if sub_key in key] #[K_50, K_60..]
                    
                if c_e['eval_method'] == 'top_k':
                    
                    for i in range(len(c_e['top_k'])):
                        if metric=='confusion_matrix':
                            conf_mat = values[i].ravel()
                            
                            for j in range(len(conf_mat)):
                                label = conf_head[j]
                                col_label = label + '_@k='+str(c_e['top_k'][i])
                                data.ix[row, col_label] = conf_mat[j] 
                            
                        else:
                            col_label = metric + '_@k='+str(c_e['top_k'][i])
                            data.loc[row, col_label] = values[i]
                
                elif c_e['eval_method'] == 'thresholding':
                    for i in range(len(c_e['thresholding'])):
                        col_label = metric + '_p>'+str(c_e['thresholding'][i])
                        data.loc[row, col_label] = values[i]
            
            #source files
            data.loc[row, 'Prediction Source'] = c_p['dir'] + c_p['file']
            data.loc[row, 'Referral Source'] = c_r['dir'] + c_r['file']
            data.loc[row, 'Result Output'] = c_e['dir'] + c_e['file']
            
            #timestamps
            data.loc[row, 'End Time'] = pd.Timestamp.now(tz='US/Eastern')
            data.loc[row, 'Total Time'] = data.loc[row, 'End Time'] - data.loc[row, 'Start Time']
            row = row + 1
    
    data['Num Samples'] = data['Num Samples'].astype(int)

    #moving the start time col to end
    cols = data.columns.tolist()
    cols = cols[1:-2] + [cols[0]] + cols[-2:]
    data = data[cols]
    
    return data