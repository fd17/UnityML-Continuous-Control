# UnityML-Navigation Project

This project is aimed at solving a provided Unity Machine Learning Agents Toolkit challenge. In this challenge, a simulated robotic agent must learn to reach with its arm for a certain position by applying torque to the joint motors. This project uses the 20-agent version of the environment.

<img src="https://github.com/fd17/UnityML-Continuous-Control/blob/master/trained_example.gif" width="480" height="270" />

## Requirements

* Windows (64 bit)
* [Python 3.6](https://www.python.org/downloads/release/python-366/)
* [Unity ML-Agents Toolkit](https://www.python.org/downloads/release/python-366/)
* [Pytorch](https://pytorch.org/)
* [Matplotlib](https://matplotlib.org/) 
* [Jupyter](http://jupyter.org/) 

## Installation
Recommended way of installing the dependencies is via [Anaconda](https://www.anaconda.com/download/). To create a new Python 3.6 environment run

`conda create --name myenv python=3.6`

Activate the environment with

`conda activate myenv`

[Click here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) for instructions on how to install the Unity ML-Agents Toolkit.

Visit [pytorch.org](https://pytorch.org/) for instructions on installing pytorch.

Install matplotlib with

`conda install -c conda-forge matplotlib`

Jupyter should come installed with Anaconda. If not, [click here](http://jupyter.org/install) for instructions on how to install Jupyter.


## Getting started
The project can be run with the provided jupyter notebooks. Reacher_Observe.ipynb allows one to observe a fully trained agent in the environment. Reacher_Training.ipynb can be used to train a new agent or continue training a pre-trained agent. Several partially pre-trained agents and one fully trained agents are stored in the savedata folder.

## Environment
The environment is a a platform with 20 robot arms. Each arm has its own target sphere hovering around it at random positions. The observation space for a single agent consists of 33 continuous variables, containing observables like position, rotation and angular velocities. The action space consists of 4 continuous actions between -1.0 and +1.0, corresponding to torque at the arm's joints. Each of the two joints can be rotated in two directions.

## Algorithm
The training algorithm used here is [Proximal Policy Optimization (PPO)](https://blog.openai.com/openai-baselines-ppo/) with clipped surrogate functions, as introduced by OpenAI in 2017. A detailed description can be found in the [2017 paper by Schuman et al.](https://arxiv.org/abs/1707.06347). As opposed to Deep Deterministic Policy Gradients, PPO works with probabilities for every action. By collecting trajectories of states, actions and rewards, it tries to increase the probability for actions that resulted in a high reward and decrease probabilities that resulted in a low reward. The original version of this algorithm is called REINFORCE, but it has several shortcomings that were partially overcome with improvements like Trust Region Policy Optimization (TRPO). PPO greatly simplifies the approch TRPO and is especially useful for distributed training where many parallel agents collect experience.

## Agent
The agent uses a basic actor-critic model. Both actor and critic are standard fully connected feed forward artificial neural networks, both with two hidden layers at 64 units each with ReLU activation. The actor predicts 4 action values and than uses them as mean for a gaussian to sample actions from. The standard deviation is an extra trainable parameter of the model. The critic is trained to estimate an advantage function.

## Training
During training, the agent collects a maximum of tmax steps of experience. Then it performs several steps of PPO on the collected trajectory using minibatches. The following hyperparameters are used during training

| parameter   | value    |  description |
|---------|---------------|-------------|
| tmax| 300 | max number of steps to collect |
| max_episodes| 800 | maximum number of episodes before quitting |
| ppo_epochs | 5 | number of PPO training epochs|
| batchsize| 500 | size of the mini-batches used during one epoch |
|discount|0.98|discount value for future returns|
|optimizer|Adam|Adam with lr=0.0003 and eps=1e-5|
|ratio_clip|0.2|r-parameter of PPO, advised value from the paper |
|max_grad_norm|0.5| maximum value of the gradient when performing optimization step|


With these settings, the agent should learn to solve the environment 
within 200 episodes.

