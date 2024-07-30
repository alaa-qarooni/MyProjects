# My Projects
This repository contains a set of my personal projects

## Animation
The Animation folder contains several python files that implement the Deep Q Learning algorithm for Reinforcement Learning that teaches a neural network to choose a speed/angle ($v$, $\theta$) action pair for a kinematic ball that will make it avoid colliding or getting too close to a set of N dynamic balls in a specified enclosed space.

The implemented DQN algorithm follows best practices in RL. Each of these are tunable hyperparameters that can improve learning:
1. _Decayed Epsilon Greedy_: The decayed epsilon greedy method ensures exploration is preferred early on in training while exploitation is preferred later on. The rate at which this happens is determined by the rate of decay in this formula: $\varepsilon_{end} + (\varepsilon_{start} - \varepsilon_{end})e^{-steps/r_{decay}}$
2. _Memory Replay_: This approach samples a random batch of past experiences to update the weights, which has been shown to decorrelate past experiences and leads to better learning.
3. _Gradient Clipping_: Prevents any "exploding" weights by clipping them to a specified maximum value.
4. _Soft Update of Weights_: Update the target network by incorporating only a fraction of the learned network weights to the target network weights, which has been shown to stabilize outputs.

### An intelligent ball!
I initially set the reward function to assign a high penalty for collisions and a reward equal to the simulation timestep when no collisions occur, but the resulting NN did not avoid collisions. I then set the reward/penalty function to account for the number and distance of dynamic balls around the kinematic ball hoping that it would recognize space where there are too many balls around and hopefully avoid it. Meaning that if there are too many dynamic balls close to our kinematic ball, the RL algoritm will assign a high penalty to the action that led it there, and vice versa when the action leads the ball to an empty region in the space. This still led to collisions, but upon closer inspection I noticed a very interesting phenomenon: the NN taught the kinematic ball to choose actions that slowly create a (relatively) empty space around it! This means that the kinematic ball would not be afraid to hit the dynamic balls around it if it means that, eventually, it will have very few balls in its vicinity. The video below shows this behavior. Around 10 seconds in, the ball shifts to the right side, and would intentionally move towards balls that it feels are encroaching on its space to push them away, and therefore keeps the space around it relatively empty!
![](https://github.com/alaa-qarooni/MyProjects/blob/main/Animation/video.gif)

### Possible Extension
While simulating after the NN is trained, I am instructing the NN to assign an action to the kinematic ball in 0.25s time increments because otherwise the motion becomes
too jittery. This results from the ball choosing sometimes contradictory actions too frequently. The simulation steps in 1/200s increments, so if one velocity-angle action is chosen as (2, 0) and the next one is (2, pi), and the next one is (3,pi/4), and so on, the motion becomes too unstable. However, this could be solved using the Proximal Policy Optimization (PPO) Algorithm, where policy updates during training are chosen so that, regardless of the penalty/reward associated, they don't lead to significant changes from the previous configuration. Using PPO might lead to better results and a smoother movement of the kinematic ball, which could be an interesting extension to this project that I might pursue at a later time.

## TRM - In-progress
This project aims to extract information from the Illinois Technical Reference Manual. This is an interesting exercise at using real world documents that are
relatively well-structured in an ML-context. The idea is to extract all the default values of variables in the TRM, including those that have multiple values
provided in tables, and all the energy, demand, gas and water savings formulas. We take the variable values and plug them into their corresponding formulas to get the
appropriate saving. An example formula with some of its variables is shown below, but you can browse the entire TRM to get a sense of its structure in the TRM folder above. The formula below is for gas savings of ENERGY STAR Clothes Washers. I am in the process of fixing my variable extraction code in the TRM_preprocessing.py file
to identify all the variables correctly, after which I am going to develop an many-to-one RNN that takes the entire variable definitions as input sequences and outputs
the correct variable definition. As a simple example, in Therm_convert below, the RNN will take "Converstion factor from kWh to Therm = 0.03412" and output 0.03412. Not
all variables have a clean value definition like this, especially the ones in tables, but once I correctly label a subsample of all the variables, I am hoping that the structured nature of the TRM will allow the trained RNN to correctly identify unlabeled ones.
![](https://github.com/alaa-qarooni/MyProjects/blob/main/TRM/example.png)

## Workout
This project takes a set of images (painstakingly!) screenshotted from my workout app and creates a table of all the workouts in the format of {_Date_: , _Exercise_: , _Instructions_: , _Result_: }. I initially tried going into the app's data files and extracting the data from there, but after consulting with my coach from the app, it seems that only he has access to such data. He was having trouble downloading it for me so I resorted to screenshotting the images and using Google's Tesseract OCR to process the image data. I have successfully completed this and you can view the output in the workout_data.csv file. The main takeaways involve needing to pre-process the images for the OCR to work since some regions in red prevented the OCR engine to process the text within them, and needing to perform parallel processing using Python's **ThreadPoolExecutor** in order to drasticallly speed up runtime.
<div align="center">
    <img src="https://github.com/alaa-qarooni/MyProjects/blob/main/Workout/images/IMG_0228.PNG" alt="drawing" width="200"/>
</div>