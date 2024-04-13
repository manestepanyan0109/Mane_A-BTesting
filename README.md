Overview

The repository consists of the following main components:

Bandit Classes: Abstract base class and concrete implementations of Epsilon Greedy and Thompson Sampling bandit algorithms.
Visualization Class: Class for visualizing experimental results using matplotlib and seaborn.
Comparison Function: Function to compare the performance of Epsilon Greedy and Thompson Sampling algorithms.
Example Usage: Demonstrates how to use the implemented algorithms and visualize the results.
Usage

To use the provided implementations:

Install the required libraries by running pip install -r requirements.txt.
Import the necessary classes and functions from the provided modules.
Define the bandit rewards and the number of trials.
Initialize instances of the bandit algorithms (EpsilonGreedy and ThompsonSampling).
Run experiments using the experiment method of the bandit instances.
Compare the performance of the algorithms using the comparison function.
Algorithms

Epsilon Greedy
The Epsilon Greedy algorithm explores with probability ε and exploits the best-known option with probability 1-ε. It maintains estimates of the rewards for each bandit and updates these estimates based on observed rewards.

Thompson Sampling
Thompson Sampling is a probabilistic algorithm that maintains a posterior distribution over the true reward probabilities of each bandit. It samples from these distributions to determine which bandit to pull at each step, with higher probability assigned to bandits with higher estimated rewards.

Visualization

The Visualization class provides methods for visualizing the convergence of average rewards over trials for a single algorithm (plot1) and comparing cumulative rewards and regrets of two algorithms (plot2).

Comparison

The comparison function compares the performance of Epsilon Greedy and Thompson Sampling algorithms in terms of total rewards and regrets. It also plots the convergence of rewards for both algorithms, both in linear and logarithmic scales.

Requirements

The code is implemented in Python 3 and requires the following libraries:

numpy
pandas
matplotlib
seaborn
scipy
