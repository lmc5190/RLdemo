# Epsilon Decay
## A simple experiment with decay rates

In plot foundation/dQ_direct_std, we see dynasarsa peak and then have a long tail throughout the rest of the experiment.
I'd like to conduct an experiment to see if we can shorten this tail by increasing the epsilon decay rate.

I plan to plot solution time versus multiple decay rates. The value of epsilon is governed by

\epsilon(t) = \max \left \{ \epsilon_{min}, \min(\epsilon_{max}, 1-\log_{10}\xi t) \right \}
