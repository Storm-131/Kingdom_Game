#---------------------------------------------------------*\
# Title: Model (OOP)
# Author: 
#---------------------------------------------------------*/
#!/usr/bin/env python3

import numpy as np
import math
from src.Kingdom import Kingdom
from utils.plotting import plot_evaluation
from tqdm import tqdm
from src.eval import evaluation_game

PRINTER = False # Print list of actions-frequencies and win-rates atfer training

class QLearningAgent:
    
    def __init__(self, total_episodes=1000, max_turns=50):
        # Discretization parameters - these could be adjusted to tune the learning process
        
        self.castle_strength_buckets = 3  # e.g., [0-20], [21-40], ..., [81-100]
        self.resource_buckets = 3  # e.g., [0-25], [26-50], [51-75], [76-100]
        self.soldier_buckets = 5  # e.g., [0-25], [26-50], [51-75], [76-100]
        
        # Actions are the same as before: Attack, Gather Resources, Train Soldiers, Fortify Castle
        self.n_actions = 4
        
        # Total number of states (Squared for both Kingdoms)
        self.n_states = (self.castle_strength_buckets * self.resource_buckets * self.soldier_buckets) ** 2
        self.q_table = np.zeros((self.n_states, self.n_actions))

        # Placeholder variables for the learning rate, discount factor, and exploration rate
        self.alpha = 0.9  # Learning rate
        self.gamma = 0.9  # Discount factor

        # Epsilon Decay Parameters
        self.epsilon_start = 1.0
        self.epsilon_min = 0.01
        self.total_episodes = total_episodes
        self.epsilon_decay_rate = math.exp((math.log(self.epsilon_min) - math.log(self.epsilon_start)) / total_episodes)

        # Further Parameter
        self.max_turns = max_turns
        self.action_names = ("Gather Resources", "Train Soldiers", "Fortify Castle", "Attack")
        
    # ---------------------------------------------------------*/
    # Functions (Helper)
    # ---------------------------------------------------------*/
    def discretize_state(self, kingdom):
        """Discretizes the state of a kingdom into predefined buckets for Q-learning. """
        castle_strength = min(int(kingdom.castle_strength / (100 / self.castle_strength_buckets)), self.castle_strength_buckets - 1)
        resources = min(int(kingdom.resources / (75 / self.resource_buckets)), self.resource_buckets - 1)
        soldiers = min(int(kingdom.soldiers / (75 / self.soldier_buckets)), self.soldier_buckets - 1)
        return castle_strength, resources, soldiers

    def get_state_index(self, state1, state2):
        """The function combines the discretized states of two kingdoms (A and B) into a single index 
        for a Q-table in a reinforcement learning environment. It scales the index of Kingdom A's state 
        by the total number of possible states for a single kingdom, then adds the index of Kingdom B's 
        state, ensuring a unique index for every possible state combination."""
        state1_index = (state1[0] * (self.resource_buckets * self.soldier_buckets) + state1[1] * self.soldier_buckets + state1[2])
        state1_index *= self.castle_strength_buckets * self.resource_buckets * self.soldier_buckets
        state2_index = (state2[0] * (self.resource_buckets * self.soldier_buckets) + state2[1] * self.soldier_buckets + state2[2])
        return state1_index + state2_index

    def choose_action(self, state_index, epsilon):
        """Function to choose an action, using epsilon-greedy policy"""
        if np.random.random() < epsilon:
            return np.random.choice([0, 1, 2, 3])
        else:
            return np.argmax(self.q_table[state_index])

    def get_reward(self, prev_state1, prev_state2, cur_state1, cur_state2, action_taken_by_kingdom1, attack_success, kingdom, opponent):
        # Define the reward values (these can be adjusted)
        reward_for_winning = 100  
        penalty_for_lack_of_rsrc = -5
        penalty_for_failed_attack = -10
        reward_for_successful_attack = 0
        reward_for_defending = 0
        reward_for_gathering = 0
        reward_for_training = 0
        reward_for_fortifying = 0

        reward = 0
        
        # Train Soldiers, but nothing changed due to lack of resources
        if action_taken_by_kingdom1 == 2:  
            if cur_state1[2] == prev_state1[2]:  # Not enough resources to train soldiers
                reward += penalty_for_lack_of_rsrc

        # Fortify Castle, but nothing changed due to lack of resources
        elif action_taken_by_kingdom1 == 3:
            if cur_state1[0] == prev_state1[0]:  # Not enough resources to fortify castle
                reward += penalty_for_lack_of_rsrc
        
        elif action_taken_by_kingdom1 == 0:  # Attack
            if not attack_success:  # Attack was not successful
                reward += penalty_for_failed_attack
    
        # Check if the game has ended with the agent's victory
        if kingdom.castle_strength > 0 and opponent.castle_strength <= 0:
            reward += reward_for_winning
            
        return reward

    def update_q_table(self, state_index, action, reward, next_state_index):
        """Update the Q-table for Q-learning based on the current state, action, and observed reward.
        
        This function applies the Q-learning update rule to modify the Q-values within the table. 
        It uses the temporal difference (TD) learning approach, adjusting the value of the current 
        state-action pair towards a target value which is a combination of the immediate reward and 
        the discounted maximum future reward in the next state.

        ## Parameters:
        - `q_table` (numpy.ndarray): The Q-table, with rows corresponding to states and columns to actions.
        - `state_index` (int): The index of the current state in the Q-table.
        - `action` (int): The action taken in the current state.
        - `reward` (float): The immediate reward received from taking the action.
        - `next_state_index` (int): The index of the next state after taking the action.
        - `alpha` (float): The learning rate, determining how much the new information affects the old Q-value.
        - `gamma` (float): The discount factor, balancing the importance of immediate and future rewards.

        ## Returns:
        None: The function updates the Q-table in place and does not return anything.
        """
        # Q-learning update rule: Q(s, a) = Q(s, a) + α * [R(s, a) + γ * max(Q(s', a')) - Q(s, a)]
        best_next_action = np.argmax(self.q_table[next_state_index])
        # R(s, a) + γ * max(Q(s', a'))
        td_target = reward + self.gamma * self.q_table[next_state_index][best_next_action]
        # R(s, a) + γ * max(Q(s', a')) - Q(s, a)
        td_delta = td_target - self.q_table[state_index][action]
        # Q(s, a) += α * (R(s, a) + γ * max(Q(s', a')) - Q(s, a))
        self.q_table[state_index][action] += self.alpha * td_delta

    #---------------------------------------------------------*/
    # Functions (Main)
    #---------------------------------------------------------*/
    def take_turn(self, kingdom, opponent, epsilon):
        """Function for a single turn in the game. It takes in the current state of the kingdom and 
        opponent, chooses an action, and updates the Q-table. It returns a boolean indicating whether
        the game is over."""
        # Store the previous states
        prev_state1 = self.discretize_state(kingdom)
        prev_state2 = self.discretize_state(opponent)
        
        state_index = self.get_state_index(prev_state1, prev_state2)

        # Kingdom takes its turn
        action = self.choose_action(self.q_table, self.state_index, self.epsilon)
        
        # Handle and execute the different actions
        attack_outcome = None
        
        if action == 0:  # Ressourcen sammeln
            kingdom.gather_resources()
            attack_outcome = None
        elif action == 1:  # Soldaten trainieren
            kingdom.train_soldiers()
        elif action == 2:  # Burg befestigen
            kingdom.fortify_castle()
        elif action == 3:
            attack_outcome, _ = kingdom.attack(opponent)

        # Store the current states
        curr_state1 = self.discretize_state(kingdom)
        curr_state2 = self.discretize_state(opponent)
        next_state_index = self.get_state_index(curr_state1, curr_state2)

        # Calculate the reward based on the action taken and the outcome
        reward = self.get_reward(prev_state1, prev_state2, curr_state1, curr_state2, action, attack_outcome, kingdom, opponent)

        # Print statement to show the action taken, the current state, and the reward
        if PRINTER:
            action_names = ["Attack", "Gather Resources", "Train Soldiers", "Fortify Castle"]
            print(f"{kingdom.name} takes action: {action_names[action]} | State: {prev_state1}, Opponent State: {prev_state2} | Reward: {reward}")

        # Update the Q-table
        self.update_q_table(self.q_table, self.state_index, self.action, self.reward, self.next_state_index, self.alpha, self.gamma)

        # Check if game is done
        return (kingdom.castle_strength <= 0 or opponent.castle_strength <= 0, reward, action)

    def run_episode(self, epsilon):
        """
        The function simulates turns between two kingdoms, `Kingdom A` and `Kingdom B`. `Kingdom A` uses the Q-table 
        for decision-making, while `Kingdom B` takes random actions. The episode ends when either a terminal state 
        is reached or the maximum number of turns is exceeded.

        Parameters:
        - `q_table` (numpy.ndarray): The Q-table, used for determining actions in Q-learning.
        - `alpha` (float, optional): The learning rate. Determines the impact of new information on the Q-values.
        - `gamma` (float, optional): The discount factor. Balances the importance of immediate and future rewards.
        - `epsilon` (float, optional): The exploration rate. Determines the probability of taking a random action for exploration.
        - `max_turns` (int, optional): The maximum number of turns for the episode. Helps in terminating the episode to avoid infinite loops.
        """
        kingdom1 = Kingdom("Kingdom A")
        kingdom2 = Kingdom("Kingdom B")
        
        done = False
        turn_count = 0
        
        while not done:
            done, _, _ = self.take_turn(kingdom1, kingdom2, epsilon)
            turn_count += 1
            
            if done or turn_count >= self.max_turns:
                break
            done, _, _ = self.take_turn(kingdom2, kingdom1, epsilon)
            turn_count += 1
            
        return self.q_table

    def train_q_learning(self):
        """
        Train a Q-learning agent over a specified number of episodes.

        Parameters:
        - `total_episodes` (int): The total number of episodes for training the agent.
        - `q_table` (numpy.ndarray): The initial Q-table to be used and updated during training.

        The function runs multiple episodes of Q-learning, updates the Q-table, and evaluates the agent's performance 
        periodically. After training, the function saves the trained Q-table in both CSV and NPY formats and plots 
        the win rate and average reward over time.
        """
        win_rates = []
        average_rewards = []
        action_freqs_total = []
        epsilon = self.epsilon_start

        for episode in tqdm(range(self.total_episodes)):
            self.run_episode(epsilon)
            epsilon = max(self.epsilon_min, self.epsilon_decay_rate * epsilon)

            if episode % 100 == 0:  # Evaluate every 100 episodes
                win_rate, avg_reward, action_freq = evaluation_game(self.q_table)
                win_rates.append(win_rate)
                average_rewards.append(avg_reward)
                action_freqs_total.append(action_freq)
                # print(f"Episode: {episode}, Win Rate: {win_rate}, Average Reward: {avg_reward}")
        
        print(win_rates) if PRINTER else None
        print(action_freqs_total) if PRINTER else None
    
        # Save the Q-table and plot results
        np.savetxt("./out/q_table.csv", self.q_table, delimiter=",")
        np.save("./out/q_table.npy", self.q_table)
        plot_evaluation(action_freqs_total, win_rates, average_rewards)

        return self.q_table

#-------------------------Notes-----------------------------------------------*\
# 
#-----------------------------------------------------------------------------*\