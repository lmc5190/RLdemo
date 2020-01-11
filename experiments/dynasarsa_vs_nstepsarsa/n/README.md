# Varying N
## A simple experiment varying method parameters

Can obtain better sample efficiency with our methods by simply changing our method parameters? Using our most sample efficienty epsilon decay multipliers, we vary n in nstepsarsa and planning steps (called n for simplicity) in dynasarsa.

![Image of runscompleted vs n](https://github.com/lmc5190/RLdemo/blob/master/experiments/dynasarsa_vs_nstepsarsa/n/plots/ncompare_nrun_solved.png)


Looking at the number of runs completed versus algorithm parameter n, we notice a bounded region of success (4 <= n <= 16) for nstepsarsa and an semi-bounded region of success (n >= 16) for dynasarsa. Note: There are no error bars since this is a value computed over all runs.




Thanks to the developers at CodeCogs https://www.codecogs.com/latex/eqneditor.php for helping me render equations. If you use their service, please acknowledge and support them!
