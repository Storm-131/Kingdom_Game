# ---------------------------------------------------------*\
# Title: Tabular Q Learning Algorithms
# Author:
# ---------------------------------------------------------*/
#!/usr/bin/env python3

import numpy as np
from src.Kingdom import Kingdom
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.plotting import plot_item_frequencies, plot_results
import math

#---------------------------------------------------------*/
# Parameters 
#---------------------------------------------------------*/

# Discretization parameters - these could be adjusted to tune the learning process
castle_strength_buckets = 3  # e.g., [0-20], [21-40], ..., [81-100]
resource_buckets = 3  # e.g., [0-25], [26-50], [51-75], [76-100]
soldier_buckets = 5  # e.g., [0-25], [26-50], [51-75], [76-100]

# Actions are the same as before: Attack, Gather Resources, Train Soldiers, Fortify Castle
n_actions = 4

# Total number of states (Squared for both Kingdoms)
n_states = (castle_strength_buckets * resource_buckets * soldier_buckets) ** 2
q_table = np.zeros((n_states, n_actions))

# Placeholder variables for the learning rate, discount factor, and exploration rate
alpha = 0.9  # Learning rate
gamma = 0.9  # Discount factor

# Epsilon Decay Parameters
epsilon_start = 1.0
epsilon_min = 0.01
total_episodes = 1000
epsilon_decay_rate = math.exp((math.log(epsilon_min) - math.log(epsilon_start)) / total_episodes)

# Further Parameter
max_turns = 50
action_names = ["Gather Resources", "Train Soldiers", "Fortify Castle", "Attack"]

# Print list of actions-frequencies and win-rates atfer training
PRINTER = False

# ---------------------------------------------------------*/
# Functions (Helper)
# ---------------------------------------------------------*/
def discretize_state(kingdom):
    """Discretizes the state of a kingdom into predefined buckets for Q-learning. """
    # Dynamic size of the buckets, as plausible for the game
    castle_strength = min(int(kingdom.castle_strength / (100 / castle_strength_buckets)),castle_strength_buckets - 1,)
    resources = min(int(kingdom.resources / (75 / resource_buckets)), resource_buckets - 1)
    soldiers = min(int(kingdom.soldiers / (75 / soldier_buckets)), soldier_buckets - 1)
    return castle_strength, resources, soldiers

def get_state_index(state1, state2):
    """The function combines the discretized states of two kingdoms (A and B) into a single index 
    for a Q-table in a reinforcement learning environment. It scales the index of Kingdom A's state 
    by the total number of possible states for a single kingdom, then adds the index of Kingdom B's 
    state, ensuring a unique index for every possible state combination."""
    # Calculate the index for state1 (Kingdom A)
    state1_index = (
        state1[0] * (resource_buckets * soldier_buckets)
        + state1[1] * soldier_buckets
        + state1[2]
    )
    # Scale the index by the total number of possible states for both kingdoms
    state1_index *= castle_strength_buckets * resource_buckets * soldier_buckets

    # Calculate the index for state2 (Kingdom B) -> Starts at index 0, if A = (0,0,0)
    state2_index = (
        state2[0] * (resource_buckets * soldier_buckets)
        + state2[1] * soldier_buckets
        + state2[2]
    )
    # Add the index for state2 to the scaled index for state1 to get the final linear index
    linear_index = state1_index + state2_index

    return linear_index


def choose_action(q_table, state_index, epsilon):
    """Function to choose an action, using epsilon-greedy policy"""
    if np.random.random() < epsilon:  # Explore: choose a random action
        return np.random.choice([0, 1, 2, 3])
    else:  # Exploit: choose the best action from Q-table
        return np.argmax(q_table[state_index])

def get_reward(prev_state1, prev_state2, cur_state1, cur_state2, action_taken_by_kingdom1, attack_success, kingdom, opponent):
    
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

def update_q_table(q_table, state_index, action, reward, next_state_index, alpha, gamma):
    """
    Update the Q-table for Q-learning based on the current state, action, and observed reward.

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
    # Debugging line
    # print(f"Updating Q-table with state: {state_index}, action: {action}")

    # Q-learning update rule: Q(s, a) = Q(s, a) + α * [R(s, a) + γ * max(Q(s', a')) - Q(s, a)]
    best_next_action = np.argmax(q_table[next_state_index])  # max(Q(s', a'))
    # R(s, a) + γ * max(Q(s', a'))
    td_target = reward + gamma * q_table[next_state_index][best_next_action]
    # R(s, a) + γ * max(Q(s', a')) - Q(s, a)
    td_delta = td_target - q_table[state_index][action]
    # Q(s, a) += α * (R(s, a) + γ * max(Q(s', a')) - Q(s, a))
    q_table[state_index][action] += alpha * td_delta


#---------------------------------------------------------*/
# Evaluate Agent
#---------------------------------------------------------*/

def evaluate_agent(q_table, num_episodes=1, max_turns=max_turns):
    """Evaluate the agent over a specified number of episodes."""
    
    total_wins = 0
    total_reward = 0
    action_frequencies = {}
    
    for _ in range(num_episodes):
        kingdom1 = Kingdom("Kingdom A")
        kingdom2 = Kingdom("Kingdom B")
        done = False
        total_reward = 0
        turn_count = 0  # Initialize turn counter
        actions_list = []  # Liste zum Speichern der Aktionen von Kingdom1
        draw = False

        while not done:
            # Kingdom 1 takes its turn by Q-Table, learning reate = 0, exploration rate = 0
            done, reward1, action = take_turn(kingdom1, kingdom2, q_table, 0, 0.9, 0)
            total_reward += reward1
            actions_list.append(action)
            turn_count += 1
            
            if done:
                break
            elif turn_count >= max_turns:
                draw = True
                break

            # Kingdom 2 takes its turn randomly, learning reate = 0, exploration rate = 1
            done, _, _ = take_turn(kingdom2, kingdom1, q_table, 0, 0.9, 0.5)
            turn_count += 1
        
        if draw: 
            total_wins += 0.5
        elif kingdom1.castle_strength > kingdom2.castle_strength:
            total_wins += 1 
            
        # Count the actions, use names as keys
        for action in actions_list:
            if action_names[action] in action_frequencies:
                action_frequencies[action_names[action]] += 1
            else:
                action_frequencies[action_names[action]] = 1

    average_reward = total_reward / num_episodes
    win_rate = total_wins / num_episodes
    
    # print("Action Frequencies: ", action_frequencies)
    
    return win_rate, average_reward, action_frequencies

#---------------------------------------------------------*/
# Functions (Main)
#---------------------------------------------------------*/
def take_turn(kingdom, opponent, q_table, alpha, gamma, epsilon):
    """Function for a single turn in the game. It takes in the current state of the kingdom and 
    opponent, chooses an action, and updates the Q-table. It returns a boolean indicating whether
    the game is over."""
    # Store the previous states
    prev_state1 = discretize_state(kingdom)
    prev_state2 = discretize_state(opponent)
    
    state_index = get_state_index(prev_state1, prev_state2)

    # Kingdom takes its turn
    action = choose_action(q_table, state_index, epsilon)
    
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
    curr_state1 = discretize_state(kingdom)
    curr_state2 = discretize_state(opponent)
    next_state_index = get_state_index(curr_state1, curr_state2)

    # Calculate the reward based on the action taken and the outcome
    reward = get_reward(prev_state1, prev_state2, curr_state1, curr_state2, action, attack_outcome, kingdom, opponent)

    # # Print statement to show the action taken, the current state, and the reward
    # action_names = ["Attack", "Gather Resources", "Train Soldiers", "Fortify Castle"]
    # print(f"{kingdom.name} takes action: {action_names[action]} | State: {prev_state1}, Opponent State: {prev_state2} | Reward: {reward}")

    # Update the Q-table
    update_q_table(q_table, state_index, action, reward, next_state_index, alpha, gamma)

    # Check if game is done
    return (kingdom.castle_strength <= 0 or opponent.castle_strength <= 0, reward, action)

# Updating the Q-learning run function with the corrected discretize_state
def run_episode(q_table, alpha=alpha, gamma=gamma, epsilon=epsilon_start, max_turns=max_turns):
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
    turn_count = 0  # Initialize turn counter
        
    while not done:
        # Kingdom 1 takes its turn
        done, _, _ = take_turn(kingdom1, kingdom2, q_table, alpha, gamma, epsilon)
        turn_count += 1

        if done or turn_count >= max_turns:
            break

        # Kingdom 2 takes its turn (More random action, no Q-Table Update)
        done, _, _ = take_turn(kingdom2, kingdom1, q_table, alpha=0, gamma=gamma, epsilon=0.6)
        turn_count += 1
            
        # print("Turn count: ", turn_count)
        
    # if turn_count >= max_turns:
        # print("Game ended due to turn limit")
        
    return q_table

#---------------------------------------------------------*/
# Training Loop
#---------------------------------------------------------*/

def train_q_learning(q_table=q_table, total_episodes=1000, epsilon=epsilon_start, alpha=alpha):
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

    # Run multiple episodes to train the Q-table
    for episode in tqdm(range(total_episodes)):
        
        
        run_episode(q_table, epsilon=epsilon_start, alpha=alpha)
        epsilon = max(epsilon_min, epsilon_decay_rate * epsilon)

        if episode % 100 == 0:  # Evaluate every 100 episodes
            win_rate, avg_reward, action_freq = evaluate_agent(q_table)
            win_rates.append(win_rate)
            average_rewards.append(avg_reward)
            action_freqs_total.append(action_freq)
            # print(f"Episode: {episode}, Win Rate: {win_rate}, Average Reward: {avg_reward}")

    print(win_rates) if PRINTER else None
    print(action_freqs_total) if PRINTER else None
    
    q_table_trained = q_table
    np.savetxt("./out/q_table.csv", q_table, delimiter=",")
    np.save("./out/q_table.npy", q_table)
    plot_item_frequencies(action_freqs_total)
    plot_results(win_rates, average_rewards)
    return q_table_trained
            
# -------------------------Notes-----------------------------------------------*\
#
# -----------------------------------------------------------------------------*\
