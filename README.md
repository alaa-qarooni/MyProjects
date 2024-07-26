# My Projects
This repository contains a set of my personal projects

## Animation
The Animation folder contains several python files that implement the Deep Q Learning algorithm for Reinforcement Learning that
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

### An intelligent ball!
I set the reward/penalty function to account for the number and distance of dynamic balls around the kinematic ball. This means that if there are too many dynamic balls close to
our kinematic ball, the RL algoritm will assign a high penalty to the state-action pair that led it there. This leads to a very interesting phenomenon: the NN taught the kinematic ball to choose
actions that slowly create a (relatively) empty space around it! This means that the kinematic ball would not be afraid to hit the dynamic balls around it if it means that,
eventually, it will have very few balls in its vicinity. The video below shows this behavior. Around 10 seconds in, the ball shifts to the right side, and would intentionally
move towards balls that it feels are encroaching on its space to push them away, and therefore keeps the space around it relatively empty!
![](https://github.com/alaa-qarooni/MyProjects/blob/main/Animation/video.gif)
