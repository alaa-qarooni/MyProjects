# My Projects
This is a repository containing my personal projects.

## Animation
The Animation folder contains several python files that implement the Deep Q Learning (DQN) algorithm for reinforcement learning (RL). The algorithm
teaches a neural network to choose a speed/angle ($v$, $\theta$) action pair for a kinematic ball that will make it avoid colliding or getting too
close to a set of N dynamic balls in a specified enclosed space.

The implemented DQN algorithm follows best practices in RL:
1. _Decayed Epsilon Greedy_: The decayed epsilon greedy method ensures exploration is preferred early on in training while exploitation is preferred
later on. The rate at which this happens is determined by the rate of decay in the following way: $\varepsilon_{end} + (\varepsilon_{start) - \varepsilon_{end})e^{-steps/r_{decay}}$
