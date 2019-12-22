import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

col_names = ['method', 'run', 'episode', 'terminal_timestep']
df = pd.DataFrame(columns=col_names)
df['terminal_timestep'] = pd.Series(dtype=np.int64)

df.loc[len(df)] = ["dynasarsa", 1, 1, 5000]
df.loc[len(df)] = ["dynasarsa", 2, 1, 4000]
df.loc[len(df)] = ["dynasarsa", 1, 2, 4000]
df.loc[len(df)] = ["dynasarsa", 2, 2, 3500]

df_wmeans = df.groupby(['method', 'episode'])['terminal_timestep'].mean().to_frame(name = 'mean_terminal_timestep').reset_index()

plt.style.use('seaborn-whitegrid')
x = df_wmeans.loc[:,"mean_terminal_timestep"].to_numpy()
y = df_wmeans.loc[:,"episode"].to_numpy()

plt.plot(x, y, 'o', color='black')
plt.show()