from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

class Bandit(ABC):
    def __init__(self, p):
        """
        Initialize a bandit with a given win rate.

        Parameters:
        p (float): The true win rate of the bandit.
        """
        self.p = p
        self.p_estimate = 0
        self.N = 0
        self.r_estimate = 0

    def __repr__(self):
        """
        Represent the bandit object as a string.
        """
        return f'Bandit with a Win Rate of {self.p}'

    @abstractmethod
    def pull(self):
        """
        Simulate pulling the bandit arm and return the reward.
        """
        pass

    @abstractmethod
    def update(self, x):
        """
        Update the bandit's estimate based on the observed reward.

        Parameters:
        x (float): The observed reward from pulling the bandit arm.
        """
        pass

    @abstractmethod
    def experiment(self):
        """
        Run the bandit algorithm and return results.
        """
        pass

    def report(self, N, results, algorithm="Epsilon Greedy"):
        """
        Generate a report of the bandit experiment results.

        Parameters:
        N (int): Number of trials in the experiment.
        results (tuple): Tuple containing experiment results.
        algorithm (str): Name of the algorithm used in the experiment.
        """
        if algorithm == 'EpsilonGreedy':
            cumulative_reward_average, cumulative_reward, cumulative_regret, bandits, chosen_bandit, reward, count_suboptimal = results
        else:
            cumulative_reward_average, cumulative_reward, cumulative_regret, bandits, chosen_bandit, reward = results

        # Write results to CSV files
        data_df = pd.DataFrame({
            'Bandit': [b for b in chosen_bandit],
            'Reward': [r for r in reward],
            'Algorithm': algorithm
        })
        data_df.to_csv(f'{algorithm}.csv', index=False)

        data_df1 = pd.DataFrame({
            'Bandit': [b for b in bandits],
            'Reward': [p.p_estimate for p in bandits],
            'Algorithm': algorithm
        })
        data_df1.to_csv(f'{algorithm}_Final.csv', index=False)

        # Print bandit statistics
        for b in range(len(bandits)):
            print(f'Bandit with True Win Rate {bandits[b].p} - Pulled {bandits[b].N} times - Estimated average reward - {round(bandits[b].p_estimate, 4)} - Estimated average regret - {round(bandits[b].r_estimate, 4)}')
            print("--------------------------------------------------")

        # Print cumulative reward and regret
        print(f"Cumulative Reward: {sum(reward)}\n")
        print(f"Cumulative Regret: {cumulative_regret[-1]}\n")
        if algorithm == 'EpsilonGreedy':
            print(f"Percent suboptimal: {round((float(count_suboptimal) / N), 4)}")

class Visualization:
    def plot1(self, N, results, algorithm='EpsilonGreedy'):
        """
        Plot the convergence of average reward over trials for a single algorithm.

        Parameters:
        N (int): Number of trials in the experiment.
        results (tuple): Tuple containing experiment results.
        algorithm (str): Name of the algorithm used in the experiment.
        """
        cumulative_reward_average = results[0]
        bandits = results[3]

        plt.plot(cumulative_reward_average, label='Cumulative Average Reward')
        plt.plot(np.ones(N) * max([b.p for b in bandits]), label='Optimal Reward')
        plt.legend()
        plt.title(f"Win Rate Convergence for {algorithm}")
        plt.xlabel("Number of Trials")
        plt.ylabel("Estimated Reward")
        plt.show()

    def plot2(self, results_eg, results_ts):
        """
        Plot the cumulative reward and regret comparison between two algorithms.

        Parameters:
        results_eg (tuple): Tuple containing experiment results for Epsilon Greedy.
        results_ts (tuple): Tuple containing experiment results for Thompson Sampling.
        """
        cumulative_rewards_eps = results_eg[1]
        cumulative_rewards_th = results_ts[1]
        cumulative_regret_eps = results_eg[2]
        cumulative_regret_th = results_ts[2]

        # Plot cumulative rewards comparison
        plt.plot(cumulative_rewards_eps, label='Epsilon-Greedy')
        plt.plot(cumulative_rewards_th, label='Thompson Sampling')
        plt.legend()
        plt.title("Cumulative Reward Comparison")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Reward")
        plt.show()

        # Plot cumulative regret comparison
        plt.plot(cumulative_regret_eps, label='Epsilon-Greedy')
        plt.plot(cumulative_regret_th, label='Thompson Sampling')
        plt.legend()
        plt.title("Cumulative Regret Comparison")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Regret")
        plt.show()

class EpsilonGreedy(Bandit):
    def __init__(self, p):
        super().__init__(p)

    def pull(self):
        """
        Simulate pulling the bandit arm and return the reward.
        """
        return np.random.randn() + self.p

    def update(self, x):
        """
        Update the bandit's estimate based on the observed reward.

        Parameters:
        x (float): The observed reward from pulling the bandit arm.
        """
        self.N += 1.
        self.p_estimate = (1 - 1.0/self.N) * self.p_estimate + 1.0/ self.N * x
        self.r_estimate = self.p - self.p_estimate

    def experiment(self, BANDIT_REWARDS, N, t=1):
        """
        Run the Epsilon Greedy experiment.

        Parameters:
        BANDIT_REWARDS (list): List of true win rates for bandits.
        N (int): Number of trials in the experiment.
        t (int): Time step in the experiment.

        Returns:
        tuple: Tuple containing experiment results.
        """
        # Initialize bandits and other variables
        bandits = [EpsilonGreedy(p) for p in BANDIT_REWARDS]
        means = np.array(BANDIT_REWARDS)
        true_best = np.argmax(means)
        count_suboptimal = 0
        EPS = 1/t
        reward = np.empty(N)
        chosen_bandit = np.empty(N)

        for i in range(N):
            p = np.random.random()

            if p < EPS:
                j = np.random.choice(len(bandits))
            else:
                j = np.argmax([b.p_estimate for b in bandits])

            x = bandits[j].pull()
            bandits[j].update(x)

            if j != true_best:
                count_suboptimal += 1

            reward[i] = x
            chosen_bandit[i] = j

            t += 1
            EPS = 1/t

        cumulative_reward_average = np.cumsum(reward) / (np.arange(N) + 1)
        cumulative_reward = np.cumsum(reward)

        cumulative_regret = np.empty(N)
        for i in range(len(reward)):
            cumulative_regret[i] = N*max(means) - cumulative_reward[i]

        return cumulative_reward_average, cumulative_reward,  cumulative_regret, bandits, chosen_bandit, reward, count_suboptimal

class ThompsonSampling(Bandit):
    def __init__(self, p):
        super().__init__(p)
        self.lambda_ = 1
        self.tau = 1

    def pull(self):
        """
        Simulate pulling the bandit arm and return the reward.
        """
        return np.random.randn() / np.sqrt(self.tau) + self.p

    def sample(self):
        """
        Sample from the bandit's posterior distribution.
        """
        return np.random.randn() / np.sqrt(self.lambda_) + self.p_estimate

    def update(self, x):
        """
        Update the bandit's estimate based on the observed reward.

        Parameters:
        x (float): The observed reward from pulling the bandit arm.
        """
        self.p_estimate = (self.tau * x + self.lambda_ * self.p_estimate) / (self.tau + self.lambda_)
        self.lambda_ += self.tau
        self.N += 1
        self.r_estimate = self.p - self.p_estimate

    def plot(self, bandits, trial):
        """
        Plot the bandit distributions after a certain number of trials.

        Parameters:
        bandits (list): List of bandit objects.
        trial (int): Current trial number.
        """
        x = np.linspace(-3, 6, 200)
        for b in bandits:
            y = norm.pdf(x, b.p_estimate, np.sqrt(1. / b.lambda_))
            plt.plot(x, y, label=f"real mean: {b.p:.4f}, num plays: {b.N}")
            plt.title("Bandit distributions after {} trials".format(trial))
        plt.legend()
        plt.show()

    def experiment(self, BANDIT_REWARDS, N):
        """
        Run the Thompson Sampling experiment.

        Parameters:
        BANDIT_REWARDS (list): List of true win rates for bandits.
        N (int): Number of trials in the experiment.

        Returns:
        tuple: Tuple containing experiment results.
        """
        bandits = [ThompsonSampling(m) for m in BANDIT_REWARDS]

        sample_points = [5, 20, 50, 200, 500, 1000, 2500, 10000, 19999]
        reward = np.empty(N)
        chosen_bandit = np.empty(N)

        for i in range(N):
            j = np.argmax([b.sample() for b in bandits])

            if i in sample_points:
                self.plot(bandits, i)

            x = bandits[j].pull()
            bandits[j].update(x)

            reward[i] = x
            chosen_bandit[i] = j

        cumulative_reward_average = np.cumsum(reward) / (np.arange(N) + 1)
        cumulative_reward = np.cumsum(reward)

        cumulative_regret = np.empty(N)

        for i in range(len(reward)):
            cumulative_regret[i] = N*max([b.p for b in bandits]) - cumulative_reward[i]

        return cumulative_reward_average, cumulative_reward,  cumulative_regret, bandits, chosen_bandit, reward

def comparison(N, results_eg, results_ts):
    """
    Compare the performance of Epsilon Greedy and Thompson Sampling algorithms.

    Parameters:
    N (int): Number of trials in the experiment.
    results_eg (tuple): Tuple containing experiment results for Epsilon Greedy.
    results_ts (tuple): Tuple containing experiment results for Thompson Sampling.
    """
    cumulative_reward_eg = results_eg[0]
    cumulative_reward_ts = results_ts[0]
    bandits_eg = results_eg[3]
    reward_eg = results_eg[5]
    reward_ts = results_ts[5]
    regret_eg = results_eg[2][-1]
    regret_ts = results_ts[2][-1]

    print(f"Total Reward Epsilon Greedy: {sum(reward_eg)}")
    print(f"Total Reward Thompson Sampling: {sum(reward_ts)}")
    print(" ")
    print(f"Total Regret Epsilon Greedy: {regret_eg}")
    print(f"Total Regret Thompson Sampling: {regret_ts}")

    sns.set_style("whitegrid")
    sns.set_palette("colorblind")

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(cumulative_reward_eg, label='Epsilon Greedy', color='blue', linestyle='-', marker='o')
    plt.plot(cumulative_reward_ts, label='Thompson Sampling', color='orange', linestyle='--', marker='s')
    plt.plot(np.ones(N) * max([b.p for b in bandits_eg]), label='Optimal Reward', color='green', linestyle='-.')
    plt.legend()
    plt.title("Comparison of Win Rate Convergence (Linear Scale)", fontsize=14)
    plt.xlabel("Number of Trials", fontsize=12)
    plt.ylabel("Estimated Reward", fontsize=12)

    plt.subplot(1, 2, 2)
    plt.plot(cumulative_reward_eg, label='Epsilon Greedy', color='blue', linestyle='-', marker='o')
    plt.plot(cumulative_reward_ts, label='Thompson Sampling', color='orange', linestyle='--', marker='s')
    plt.plot(np.ones(N) * max([b.p for b in bandits_eg]), label='Optimal Reward', color='green', linestyle='-.')
    plt.legend()
    plt.title("Comparison of Win Rate Convergence (Log Scale)", fontsize=14)
    plt.xlabel("Number of Trials", fontsize=12)
    plt.ylabel("Estimated Reward", fontsize=12)
    plt.xscale("log")

    plt.tight_layout()
    plt.show()

# Example usage:
BANDIT_REWARDS = [1, 2, 3, 4]
N = 20000

eg = EpsilonGreedy(0.1)
ts = ThompsonSampling(0.1)
results_eg = eg.experiment(BANDIT_REWARDS, N)
results_ts = ts.experiment(BANDIT_REWARDS, N)

comparison(N, results_eg, results_ts)
