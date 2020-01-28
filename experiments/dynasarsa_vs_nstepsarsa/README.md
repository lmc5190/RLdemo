## Experiment
Compare an optimized dyanasarsa vs nstepsarsa for the 10x10 gym-maze environment.
Our solution condition is the maze must complete in optimal number of steps, which is 62 steps, 10 epsides in a row.
We performed a grid search over the following parameter values

Please see the READMEs in the file directories to see more details on each parameter as well as the code for the gridsearch.

## Results
1. We did not get an optimal result for nstepsarsa.
1. Dynasarsa had optimal performance, solving the maze in 14 episodes (had optimal policy by end of episode 3) with following parameters:
  * planning steps = 16, Xi = 4 for alpha decay, Xi = 32 for epsilon decay
