# Alpha Decay
## Summary of Results
1. We succeeded in finding optimal value for nstep sarsa, but found one for dynasarsa.
1. Xi = 4 for alpha decay in dynasarsa was min value.

## A simple experiment with decay rates

I'm going to plot solution episode and number of runs solved for alpha decay values (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0) for 10by10 gym maze environment averaged over 30 runs. 

Alpha decay are defined by the same eqn as [epsilon decay](https://github.com/lmc5190/RLdemo/blob/master/experiments/dynasarsa_vs_nstepsarsa/epsilon_decay/README.md) 

![equation](https://latex.codecogs.com/gif.latex?\alpha(t)&space;=&space;max(\alpha_{min}&space;,&space;min(1&space;-&space;\log_{10}\xi&space;t)))

The hyperparameters are set at optimized values for n and Xi_epsilon (see those experiments)
* n = 4 for nstepsarsa, n = 16 planning steps for dynasarsa
* Xi for epsilon is 4 for nstepsarsa, and 32 for dynasrasa

![Image of nrunsolved](https://github.com/lmc5190/RLdemo/blob/master/experiments/dynasarsa_vs_nstepsarsa/alpha_decay/plots/opt_epsdecay_n.csv_nrun_solved.png)

The plot shows that no values solved nstepsarsa, while several values 0.25 thru 2 solve dynasarsa.



![Image of alpha decay fix n and eps decay](https://github.com/lmc5190/RLdemo/blob/master/experiments/dynasarsa_vs_nstepsarsa/alpha_decay/plots/opt_epsdecay_n.csv_solution_episode.png)

Since nstepsarsa didn't solve, we only care about dynasarsa. Dynasarsa shows minimum success number at 14 episodes for Xi = 4.


Thanks to the developers at CodeCogs https://www.codecogs.com/latex/eqneditor.php for helping me render equations. If you use their service, please acknowledge and support them!
