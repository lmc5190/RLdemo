# Varying N
## A simple experiment varying method parameters

Can obtain better sample efficiency with our methods by simply changing our method parameters? Using our most sample efficienty epsilon decay multipliers, we vary n in nstepsarsa and planning steps (called n for simplicity) in dynasarsa.

![Image of runscompleted vs n](https://github.com/lmc5190/RLdemo/blob/master/experiments/dynasarsa_vs_nstepsarsa/n/plots/ncompare_nrun_solved.png)


Looking at the number of runs completed versus algorithm parameter n, we notice a bounded region of success (4 <= n <= 16) for nstepsarsa and an semi-bounded region of success (n >= 16) for dynasarsa. Note: There are no error bars since this is a value computed over all runs. The semi-bounded region for dynasarsa is expected, since more planning steps are just doing more 1-step sarsa updates on prior data. The bounded region in nstepsarsa could support correspond with the geometry of the maze and how many steps are between branching points. Further experimentation could be done on that.

We also noticed that the algorithm paramters, although labeled the same, are incomparible (n=10 for nstepsarsa is not comparible to n=10 for dyansarsa).

Next, we vary the solution episode versus the algorithm parameter and find an extremely interesting result.

![Image of solutionep vs n](https://github.com/lmc5190/RLdemo/blob/master/experiments/dynasarsa_vs_nstepsarsa/n/plots/ncompare_solution_episode.png)

The solution episode is flat across the model parameters (dynasrasa = 33, nstepsarsa = 117). This is very different from what we saw in the case of the epsilon decay rate. Perhaps, the epsilon decay rate has an overriding influence on sample efficiency compared to the model parameter. We also must consider a few other things before concluding this: the alpha decay, the max/min epsilon values and the max/min alpha values.

Note in the above plot, the error region is so small you cannot see it (about 0.1 - 0.3 of an episode). I have plotted the std errors below for proof. From this plot, we notice that dynasarsa is a bit more consistent.

![Image of stderrors_insltnep vs n](https://github.com/lmc5190/RLdemo/blob/master/experiments/dynasarsa_vs_nstepsarsa/n/plots/ncompare_stderror_solution_episode.png)


Thanks to the developers at CodeCogs https://www.codecogs.com/latex/eqneditor.php for helping me render equations. If you use their service, please acknowledge and support them!
