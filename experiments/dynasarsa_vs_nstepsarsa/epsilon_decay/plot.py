import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_meansandstderrors_overruns(df, metric):
    #returns dataframe with means and standard errors over runs for specified metric
    #grouping done by methods and decay_multiplier
    df_wmeans = df.groupby(['method','decay_multiplier'])[metric].mean().to_frame(name = 'mean_'+ metric).reset_index()
    df_wsems = df.groupby(['method', 'decay_multiplier'])[metric].sem().to_frame(name = 'stderror_'+ metric).reset_index()
    df_wmeans['stderror_'+ metric]= df_wsems.loc[:, ('stderror_'+ metric)]
    return df_wmeans

def extract_axisvalues_bymethod(df,method_name, col_name):
    #returns vector represeting axis values from dataframe containing column 'col_name' for specified 'method_name'
    return df[df['method']==method_name].loc[:,col_name].to_numpy()

def remove_unsolved_runs(df):
    #will remove any row where solution_episode = -1, so episodic information is removed prior to location of solved_ep
    #this is okay since we are aggregating a single value per run over all runs, not over episodes
    return df[df['solution_episode'] != -1]

def keep_onerow_perrun(df):
    return df.groupby(['method','run', 'decay_multiplier']).first().reset_index()

def plot_solutionepisode_vs_decayrate(df):
    df=remove_unsolved_runs(df)
    df=keep_onerow_perrun(df)
    df=compute_meansandstderrors_overruns(df, metric)
    x1= extract_axisvalues_bymethod(df,method1,'decay_multiplier')
    y1= extract_axisvalues_bymethod(df,method1,('mean_'+metric))
    error1= extract_axisvalues_bymethod(df,method1,('stderror_'+metric))
    x2= extract_axisvalues_bymethod(df,method2,'decay_multiplier')
    y2 =extract_axisvalues_bymethod(df,method2,('mean_'+metric))
    error2 =extract_axisvalues_bymethod(df,method2,('stderror_'+metric))
    plt.style.use('seaborn-whitegrid')
    plt.xlabel('decay multiplier')
    plt.ylabel('solution episode') 
    plt.plot(x1, y1, 'k-')
    plt.fill_between(x1, y1-error1, y1+error1)
    #plt.plot(x2, y2, color='yellow', 'k--')
    plt.plot(x2, y2, 'r--')
    plt.fill_between(x2, y2-error2, y2+error2)
    plt.legend(['dynasarsa', 'nstepsarsa'], loc=legend_loc)
    #plt.xlim(0,75)
    plt.savefig(figure_file, format="png")
    plt.show()
    return None

def plot_nrunsolved_vs_decayrate(df):
    #Its been observed that, for the same decay rate, sometimes the run is solved, but sometimes not
    #This plot will indicate how many solved runs per decay rate
    run_df = df.groupby(['method','run','decay_multiplier'])['solution_episode'].max().to_frame(name = 'max_solution_episode').reset_index()
    run_df['run_solved'] = np.where(run_df['max_solution_episode'] > -1, 1, 0)
    cnt_df = run_df.groupby(['method','decay_multiplier'])['run_solved'].sum().to_frame(name = 'nrun_solved').reset_index()
    
    x1= extract_axisvalues_bymethod(cnt_df,method1,'decay_multiplier')
    y1= extract_axisvalues_bymethod(cnt_df,method1,'nrun_solved')
    x2= extract_axisvalues_bymethod(cnt_df,method2,'decay_multiplier')
    y2 =extract_axisvalues_bymethod(cnt_df,method2,'nrun_solved')
    plt.style.use('seaborn-whitegrid')
    plt.xlabel('decay multiplier')
    plt.ylabel('Nruns with maze solved (out of 30)') 
    plt.plot(x1, y1, 'k-')
    plt.plot(x2, y2, 'r--')
    plt.legend(['dynasarsa', 'nstepsarsa'], loc=legend_loc)
    #plt.xlim(0,75)
    plt.savefig(figure_file, format="png")
    plt.show()
    return None

metric= 'nrun_solved'
legend_loc= 'center right' 
#best        upper right        upper left        lower left        lower right        right
#center left        center right        lower center        upper center        center
figurefile_header= "plots/10step_" 
figure_file= figurefile_header + metric + ".png"
col_names = ['method', 'run', 'episode', 'terminal_timestep', 'G_direct_mean', 'G_direct_std', 'G_direct_n',\
            'G_indirect_mean', 'G_indirect_std', 'G_indirect_n', 'dQ_direct_mean', 'dQ_direct_std', \
            'dQ_indirect_mean', 'dQ_indirect_std', 'alpha', 'epsilon', 'solution_episode', 'decay_multiplier']
data_file="data/neq10.csv"
method1='dynasarsa'
method2='nstepsarsa'

df = pd.read_csv(data_file)
df.columns = col_names

#choose appropriate function, comment out others
plot_nrunsolved_vs_decayrate(df)
#plot_solutionepisode_vs_decayrate(df)