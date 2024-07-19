# My Projects
This is a repository containing my personal projects.

## Animation
The Animation folder contains several python files that implement the Deep Q Learning (DQN) algorithm for reinforcement learning (RL). The algorithm
teaches a neural network to choose a speed/angle ($v$, $\theta$) action pair for a kinematic ball that will make it avoid colliding or getting too
close to a set of N dynamic balls in a specified enclosed space.

The implemented DQN algorithm follows best practices in RL. Each of these are tunable hyperparameters that can improve learning:
1. _Decayed Epsilon Greedy_: The decayed epsilon greedy method ensures exploration is preferred early on in training while exploitation is preferred
later on. The rate at which this happens is determined by the rate of decay in this formula: $\varepsilon_{end} + (\varepsilon_{start} - \varepsilon_{end})e^{-steps/r_{decay}}$
2. _Memory Replay_: This approach samples a random batch of past experiences to update the weights, which has been shown to decorrelate past experiences
and leads to better learning
3. _Gradient Clipping_: Prevents an "exploding" weights by clipping them to a specified maximum value.
4. _Soft Update of Weights_: Update the target network by incorporating only a fraction of the learned network weights to the target network weights, which has been shown to
stabilize outputs.
