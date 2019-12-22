import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

col_names = ['method', 'run', 'episode', 'terminal_timestep']
df = pd.DataFrame(columns=col_names)
df['episode'] = pd.Series(dtype=np.int64)
df['terminal_timestep'] = pd.Series(dtype=np.int64)

df.loc[len(df)] = ["dynasarsa", 1, 1, 5000]
df.loc[len(df)] = ["dynasarsa", 2, 1, 4900]
df.loc[len(df)] = ["dynasarsa", 3, 1, 4800]
df.loc[len(df)] = ["dynasarsa", 4, 1, 5100]
df.loc[len(df)] = ["dynasarsa", 5, 1, 5200]
df.loc[len(df)] = ["dynasarsa", 1, 2, 4000]
df.loc[len(df)] = ["dynasarsa", 2, 2, 3900]
df.loc[len(df)] = ["dynasarsa", 3, 2, 3800]
df.loc[len(df)] = ["dynasarsa", 4, 2, 4100]
df.loc[len(df)] = ["dynasarsa", 5, 2, 4200]
df.loc[len(df)] = ["nstepsarsa", 1, 1, 6000]
df.loc[len(df)] = ["nstepsarsa", 2, 1, 6100]
df.loc[len(df)] = ["nstepsarsa", 3, 1, 6200]
df.loc[len(df)] = ["nstepsarsa", 4, 1, 4900]
df.loc[len(df)] = ["nstepsarsa", 5, 1, 4800]
df.loc[len(df)] = ["nstepsarsa", 1, 2, 5000]
df.loc[len(df)] = ["nstepsarsa", 2, 2, 4500]
df.loc[len(df)] = ["nstepsarsa", 3, 2, 4900]
df.loc[len(df)] = ["nstepsarsa", 4, 2, 5100]
df.loc[len(df)] = ["nstepsarsa", 5, 2, 5200]

df_wmeans = df.groupby(['method', 'episode'])['terminal_timestep'].mean().to_frame(name = 'mean_terminal_timestep').reset_index()
df_wsems = df.groupby(['method','episode'])['terminal_timestep'].sem().to_frame(name = 'stderror_terminal_timestep').reset_index()
df_wmeans['stderror_terminal_timestep']= df_wsems.loc[:, "stderror_terminal_timestep"]

x1 = df_wmeans[df_wmeans['method']=='dynasarsa'].loc[:,"episode"].to_numpy()
y1 = df_wmeans[df_wmeans['method']=='dynasarsa'].loc[:,"mean_terminal_timestep"].to_numpy()
error1 = df_wmeans[df_wmeans['method']=='dynasarsa'].loc[:, "stderror_terminal_timestep"].to_numpy()

x2 = df_wmeans[df_wmeans['method']=='nstepsarsa'].loc[:,"episode"].to_numpy()
y2 = df_wmeans[df_wmeans['method']=='nstepsarsa'].loc[:,"mean_terminal_timestep"].to_numpy()
error2 = df_wmeans[df_wmeans['method']=='nstepsarsa'].loc[:, "stderror_terminal_timestep"].to_numpy()

plt.style.use('seaborn-whitegrid')
plt.plot(x1, y1, 'k-')
plt.fill_between(x1, y1-error1, y1+error1)
plt.plot(x2, y2, 'k--')
plt.fill_between(x2, y2-error2, y2+error2)
plt.legend(['dynasarsa', 'nstepsarsa'], loc='upper right')
plt.draw()
