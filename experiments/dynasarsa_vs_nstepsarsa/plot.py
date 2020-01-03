import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_meansandstderrors_overruns(df, metric):
    #returns dataframe with means and standard errors over runs for specified metric
    #grouping done by methods and episodes
    df_wmeans = df.groupby(['method', 'episode'])[metric].mean().to_frame(name = 'mean_'+ metric).reset_index()
    df_wsems = df.groupby(['method','episode'])[metric].sem().to_frame(name = 'stderror_'+ metric).reset_index()
    df_wmeans['stderror_'+ metric]= df_wsems.loc[:, ('stderror_'+ metric)]
    return df_wmeans

def extract_axisvalues_bymethod(df,method_name, col_name):
    #returns vector represeting axis values from dataframe containing column 'col_name' for specified 'method_name'
    return df[df['method']==method_name].loc[:,col_name].to_numpy()

metric= 'epsilon'
legend_loc= 'upper right' 
#best        upper right        upper left        lower left        lower right        right
#center left        center right        lower center        upper center        center
figurefile_header= "plots/10step_" 
figure_file= figurefile_header + metric + ".png"
col_names = ['method', 'run', 'episode', 'terminal_timestep', 'G_direct_mean', 'G_direct_std', 'G_direct_n',\
            'G_indirect_mean', 'G_indirect_std', 'G_indirect_n', 'dQ_direct_mean', 'dQ_direct_std', \
            'dQ_indirect_mean', 'dQ_indirect_std', 'alpha', 'epsilon']
data_file="data/latest.csv"
method1='dynasarsa'
method2='nstepsarsa'

df = pd.read_csv(data_file)
df.columns = col_names

#custom metric


df= compute_meansandstderrors_overruns(df, metric)

x1= extract_axisvalues_bymethod(df,method1,'episode')
y1= extract_axisvalues_bymethod(df,method1,('mean_'+metric))
error1= extract_axisvalues_bymethod(df,method1,('stderror_'+metric))
x2= extract_axisvalues_bymethod(df,method2,'episode')
y2 =extract_axisvalues_bymethod(df,method2,('mean_'+metric))
error2 =extract_axisvalues_bymethod(df,method2,('stderror_'+metric))

plt.style.use('seaborn-whitegrid')
plt.plot(x1, y1, 'k-')
plt.fill_between(x1, y1-error1, y1+error1)
plt.plot(x2, y2, 'k--')
plt.fill_between(x2, y2-error2, y2+error2)
plt.legend(['dynasarsa', 'nstepsarsa'], loc=legend_loc)
#plt.xlim(0,75)
plt.savefig(figure_file, format="png")
plt.show()