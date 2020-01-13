# Epsilon Decay
## Summary of Results
1. For solved runs, the higher the decay multiplier the higher the sample efficiency for dynasarsa, nstepsarsa had a minimum.
1. However, the higher decay multipliers have a lower chance of solving runs. This intuitively makes sense since decreasing the exploration rate too fast will prevent the agent from appropriately exploring the state space.
1. For this experiment, the most stable decay rates where maze is solved in every run is:
   1. Xi = 32 for dynasarsa  (n=10)
   1. Xi = 4 for nstepsarsa (n=10)

## A simple experiment with decay rates

In plot dQ_direct_std, we see dynasarsa peak and then have a long tail throughout the rest of the experiment.
![Image of dQdirect_std](https://github.com/lmc5190/RLdemo/blob/master/experiments/dynasarsa_vs_nstepsarsa/foundation/plots/10step_dQ_direct_std.png)

I'd like to conduct an experiment to see if we can shorten this tail by increasing the epsilon decay rate.

I plan to plot solution time versus multiple decay rates. The choose the behavior below to govern the decay of epsilon, where t is equal to the episode number and Xi is the decay rate.

![equation](https://latex.codecogs.com/gif.latex?\epsilon(t)&space;=&space;\max&space;(&space;\epsilon_{min},&space;\min(\epsilon_{max},&space;1-\log_{10}\xi&space;t)&space;))

This behavior is shown graphically looks like the plot shown for epsilon versus episode number

![Image of Epsilon](https://github.com/lmc5190/RLdemo/blob/master/experiments/dynasarsa_vs_nstepsarsa/foundation/plots/10step_epsilon.png)

I varied Xi and observe how many runs of the 10 by 10 maze are solved by dynasarsa (planning steps = 10) and nstepsarsa (n=10).

![Image of Xi Variation](https://github.com/lmc5190/RLdemo/blob/master/experiments/dynasarsa_vs_nstepsarsa/epsilon_decay/plots/10step_nrun_solved.png)

Notice that dynasarsa (planning steps = 10) could handle all tried decay rates as it solved all 30 runs. Nstepsarsa (n=10) could more than 4x-8x the baseline value (Xi = 8) yielding multiple unsolved runs for Xi > 8.

Given the sharper epsilon decay for dynasara, how much can we improve the sample efficiency? Below, I plot the solution episode number averaged over 30 runs (notice that standard error is so small, it does not even render on the graph) for each method given different epsilon decay rates. Not that any runs that did not solve the maze were omitted from the average.

![Image of Xi Variation](https://github.com/lmc5190/RLdemo/blob/master/experiments/dynasarsa_vs_nstepsarsa/epsilon_decay/plots/10step_solution_episode.png)

Here we say that as the decay rate goes up, solved puzzles have much higher sample efficiency. However, no higher decay rates are acceptable for nstepsarsa (n=10). Since dynasarsa (planning steps = 10) can handle a 4x higher eploration decay, we can confidently say that would yield a solved maze in ~35 episodes over ~105 episode solution for Xi = 1 (or a increase in sample efficieny by 3x).

It would be interesting to test different paramters for dynasara and nstepsarsa, and see if this sensitivity remains.


Thanks to the developers at CodeCogs https://www.codecogs.com/latex/eqneditor.php for helping me render equations. If you use their service, please acknowledge and support them!
