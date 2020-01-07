# Epsilon Decay
## A simple experiment with decay rates

In plot foundation/dQ_direct_std, we see dynasarsa peak and then have a long tail throughout the rest of the experiment.
I'd like to conduct an experiment to see if we can shorten this tail by increasing the epsilon decay rate.

I plan to plot solution time versus multiple decay rates. The choose the behavior below to govern the decay of epsilon, where t is equal to the episode number and Xi is the decay rate.

![equation](https://latex.codecogs.com/gif.latex?\epsilon(t)&space;=&space;\max&space;(&space;\epsilon_{min},&space;\min(\epsilon_{max},&space;1-\log_{10}\xi&space;t)&space;))

This behavior is shown graphically looks like the plot shown for epsilon versus episode number

![Image of Epsilon](https://github.com/lmc5190/RLdemo/blob/master/experiments/dynasarsa_vs_nstepsarsa/foundation/plots/10step_epsilon.png)

We  varied Xi and observe how many runs of the 10 by 10 maze are solved by dynasarsa (planning steps = 10) and nstepsarsa (n=10).

![Image of Xi Variation](https://github.com/lmc5190/RLdemo/blob/master/experiments/dynasarsa_vs_nstepsarsa/epsilon_decay/plots/10step_nrun_solved.png)

Thanks to the developers at CodeCogs https://www.codecogs.com/latex/eqneditor.php for helping me render equations. If you use their service, please acknowledge and support them!
