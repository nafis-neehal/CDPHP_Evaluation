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

def process_evaluation_data(c_p, c_r, c_e, c_visual, eval_date, all_model_evaluations):

    #result file
    result_file = c_e['dir'] + c_e['file']
        
    data = generate_tabular_data(c_p, c_r, c_e, c_visual, eval_date, all_model_evaluations)
    
    if c_e['save_csv'] == True:
        if os.path.exists(result_file) and c_e['append_csv']==True:
            data.to_csv(result_file, index = False, header=False, float_format= '%8.5f', mode='a')
        else:
            data.to_csv(result_file, index = False, float_format= '%8.5f', mode='w')
        if c_e['aws'] == True:
            Aws.upload_to_aws(result_file, c_e['bucket'], result_file)




def present_evaluation(c_e, c_visual):

    #result fileca
    result_file = c_e['dir'] + c_e['file']
    
    #clear if there is already any result file previously (from previous run/experiment), 
    #otherwise it will just keep appending, as opening in append mode
        
    data = pd.read_csv(c_e['dir'] + c_e['file'])

    #list of all unique eval dates from evaluation data result
    dates = sorted(data["Eval Date"].unique())
    date_list = [str(i) for i in dates]

    #seperate sources of prediction data (male/female)
    #first males will show, then female
    sources = sorted(data['Prediction Source'].unique())

    #show agg data
    show_agg_data(data)
    
    #show table
    if c_visual['table']['show'] == True:
        pd.set_option('display.max_columns', 100)
        pd.set_option('display.max_rows', 100)
        pd.options.display.float_format = '{:8.5f}'.format
        display(data)

    #label counter
    comparison_source_counter = 0
    distribution_source_counter = 0

    #for all eval dates
    for source in sources:
        for eval_date in date_list:
            c_e['eval_date'] = eval_date

            if(c_visual['model_comparison']['plot'] == True or c_visual['model_comparison']['save'] == True):
                generate_comparison_plot(c_e, data[data['Prediction Source']==source], c_visual, eval_date, source, comparison_source_counter)

        if (c_visual['probability_distribution']['plot'] == True or c_visual['probability_distribution']['save'] == True):
            generate_probability_distribution_plot(source, c_visual, distribution_source_counter)

        comparison_source_counter += 1
        distribution_source_counter += 1

def show_agg_data(data):
    precision_column_labels = [cols for cols in data.columns.to_list() if 'precision' in cols]
    accuracy_column_labels = [cols for cols in data.columns.to_list() if 'accuracy' in cols]
    table = pd.pivot_table(data, values = precision_column_labels + accuracy_column_labels, index = ['Model', 'Prediction Source'])
    display(table)



    
#name of models, windows, top_ks/threshold
def generate_confusion_matrix_plot(c_e, all_model_evaluations, model, window, thres, eval_method, eval_date):
    pass 
    # confusion_matrix = all_model_evaluations[model].confusion_matrix
    # win = '[' + str(window[0]) + ',' + str(window[1]) + ']'
    # th = str(thres)
    # for keys in confusion_matrix.keys():
    #     if win in keys and th in keys and eval_method in keys:
    #         conf_mat = confusion_matrix[keys]
    #         conf_mat = pd.DataFrame(conf_mat)
    #         akws = {"ha": 'left',"va": 'top'}
    #         plt.figure()
    #         ax = sns.heatmap(conf_mat, annot=True, fmt="d", annot_kws=akws, cmap='Greens_r')
    #         ax.set_title("Confusion Matrix of " + model + ' for window ' + win + " on eval_date " + str(eval_date) + ' for ' + eval_method + '_@' + str(thres))

#input: c_p, list_of_models
#output: probability distribution plots
def generate_probability_distribution_plot(file, c_visual, distribution_source_counter): 

    pred = pd.read_csv(file)
    models = c_visual['probability_distribution']['models']

    for each_model in models:
        plt.figure()
        ax = sns.distplot(pred[each_model])
        ax.set_title(each_model + ", Source: " + file)
        
        if c_visual['probability_distribution']['save'] == True:
            os.makedirs(os.path.dirname(c_visual['probability_distribution']['dir']), exist_ok=True)
            plt.savefig(c_visual['probability_distribution']['dir'] + each_model + "_source_" + str(distribution_source_counter)+".png")

        if c_visual['probability_distribution']['plot'] != True:
            plt.close();

#generate comparison plot -- for each window for each value of threshold/top-k for all models for selected metrics
def generate_comparison_plot(c_e, data, c_visual, eval_date, source, comparison_source_counter): 

    plot_window = c_e['eval_windows']
    metrics = c_visual['model_comparison']['metrics']

    #eval_method mapping
    if c_e['eval_method'] == 'thresholding':
        eval_methods = ['thresholding']
    elif c_e['eval_method'] == 'top_k':
        eval_methods = ['top_k']
    elif c_e['eval_method'] == 'both':
        eval_methods = ['top_k', 'thresholding']

    #run comparison plot
    for window in plot_window:
        
        win = Helper.window_to_range(c_e, window)
        new_data = data[data['Window']==win]

        for eval_method in eval_methods:
            for t in c_e[eval_method]:
                metric_col = [col for col in new_data.columns if str(t) in col] #original column names for metrics (All 8 metric columns)
                final_original_columns = ['Model']
                final_modified_columns = ['Model']

                for metric_name in metrics:
                    final_original_columns = final_original_columns + [col for col in metric_col if metric_name in col] #only columns passed
                    final_modified_columns = final_modified_columns + [metric_name for col in metric_col if metric_name in col]
                
                common_columns = list(set(metrics).difference(set(final_modified_columns)))

                final_original_columns = final_original_columns + common_columns
                final_modified_columns = final_modified_columns + common_columns

                pf = new_data[final_original_columns].copy()
                
                for i in range(len(pf.columns)):
                    pf.rename(columns={final_original_columns[i]:final_modified_columns[i]}, inplace=True)

                #pivot the data to covert wide dataframe to long dataframe
                pf_new = pd.melt(pf, id_vars=['Model'], value_vars=final_modified_columns[1:])

                #in seaborn version 0.9.0 and higher, it is sns.catplot(x,y,hue,data,kind,height,aspect)
                #in seaborn version 0.8.1 it is sns.factorplot(x,y,hue,data,kind,size,aspect) 
                #factorplot is categorical plot on a facetgrid
                if sns.__version__ == '0.8.1':
                    data_plot = sns.factorplot(x="variable", y="value", hue="Model", data=pf_new, kind="bar", size = 4, aspect = 3)
                else:
                    data_plot = sns.catplot(x="variable", y="value", hue="Model", data=pf_new, kind="bar", height = 4, aspect = 3)

                data_plot.set_xlabels("Metrics")
                data_plot.set_ylabels("Score")
                title = "During " + win + " on eval_date " + str(eval_date) + " with " + eval_method + "_@" + str(t) + " from Prediction Source: " + source
                data_plot.fig.suptitle(title)

                if c_visual['model_comparison']['save'] == True:
                    os.makedirs(os.path.dirname(c_visual['model_comparison']['dir']), exist_ok=True)
                    plt.savefig(c_visual['model_comparison']['dir'] + str(eval_date)+"_"+ win + "_" + eval_method + "_" + str(t) + "_source_" + str(comparison_source_counter)+".png")

                if c_visual['model_comparison']['plot'] != True:
                    plt.close();


#takes all info, returns dataframe 
def generate_tabular_data(c_p, c_r, c_e, c_visual, eval_date, all_model_evaluations):
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
            
            for metric in metric_name:
                
                values  = [value for key, value in all_model_evaluations[model].__dict__[metric].items() if sub_key in key] #[K_50, K_60..]

                if metric=='log_loss' or metric == 'brier_score_loss' or metric == 'roc_auc_score':
                    data.loc[row, metric] = values[0]
                    continue                
                
                #-----------------------------------------------------------------------------
                if c_e['eval_method'] == 'top_k':
                    
                    for i in range(len(c_e['top_k'])):
                        if metric=='confusion_matrix':
                            conf_mat = values[i].ravel()
                            
                            for j in range(len(conf_mat)):
                                label = conf_head[j]
                                col_label = label + '_@k='+str(c_e['top_k'][i])
                                data.loc[row, col_label] = conf_mat[j] 
                            
                        else:
                            col_label = metric + '_@k='+str(c_e['top_k'][i])
                            data.loc[row, col_label] = values[i]
                
                #-----------------------------------------------------------------------------
                elif c_e['eval_method'] == 'thresholding':

                    for i in range(len(c_e['thresholding'])):
                        if metric=='confusion_matrix':
                            conf_mat = values[i].ravel()
                            
                            for j in range(len(conf_mat)):
                                label = conf_head[j]
                                col_label = label + '_@p>='+str(c_e['thresholding'][i])
                                data.loc[row, col_label] = conf_mat[j] 
                            
                        else:
                            col_label = metric + '_@p>='+str(c_e['thresholding'][i])
                            data.loc[row, col_label] = values[i]
                
                #-----------------------------------------------------------------------------
                elif c_e['eval_method'] == 'both':

                    for i in range(len(c_e['top_k'])):
                        if metric=='confusion_matrix':
                            conf_mat = values[i].ravel()

                            for j in range(len(conf_mat)):
                                    label = conf_head[j]
                                    col_label = label + '_@k='+str(c_e['top_k'][i])
                                    data.loc[row, col_label] = conf_mat[j] 
                        else:
                            col_label = metric + '_@k='+str(c_e['top_k'][i])
                            data.loc[row, col_label] = values[i]


                    for i in range(len(c_e['top_k']), len(c_e['top_k'] + c_e['thresholding'])):
                        if metric=='confusion_matrix':
                            conf_mat = values[i].ravel()
                            
                            for j in range(len(conf_mat)):
                                label = conf_head[j]
                                col_label = label + '_@p>='+str(c_e['thresholding'][i-2])
                                data.loc[row, col_label] = conf_mat[j] 
                            
                        else:
                            col_label = metric + '_@p>='+str(c_e['thresholding'][i-2])
                            data.loc[row, col_label] = values[i]

            
            #source files
            data.loc[row, 'Prediction Source'] = c_p['file']
            data.loc[row, 'Referral Source'] = c_r['file']
            data.loc[row, 'Result Output'] = c_e['file']
            
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