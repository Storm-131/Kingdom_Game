#---------------------------------------------------------*\
# Title: Evaluate Agent
# Author: 
#---------------------------------------------------------*/
#!/usr/bin/env python3

import tqdm
from src import Kingdom
from utils.plotting import plot_evaluation # plot_item_frequencies, plot_results

def evaluation_game(q_table, model):
    """Evaluate the agent over a specified number of episodes."""
    
    total_wins = 0
    total_reward = 0
    action_frequencies = {}
    
    kingdom1 = Kingdom("Kingdom A")
    kingdom2 = Kingdom("Kingdom B")
    done = False
    total_reward = 0
    turn_count = 0  # Initialize turn counter
    actions_list = []  # Liste zum Speichern der Aktionen von Kingdom1
    draw = False

    while not done:
        # Kingdom 1 takes its turn by Q-Table, learning reate = 0, exploration rate = 0
        done, reward1, action = model.take_turn(kingdom1, kingdom2, model.q_table, 0, 0.9, 0)
        total_reward += reward1
        actions_list.append(action)
        turn_count += 1
        
        if done:
            break
        elif turn_count >= model.max_turns:
            draw = True
            break

        # Kingdom 2 takes its turn half randomly, learning reate = 0, exploration rate = 0.5
        done, _, _ = model.take_turn(kingdom2, kingdom1, model.q_table, 0, 0.9, 0.1)
        turn_count += 1
    
    if draw: 
        total_wins += 0.5
    elif kingdom1.castle_strength > kingdom2.castle_strength:
        total_wins += 1 
        
    # Count the actions, use names as keys
    for action in actions_list:
        if model.action_names[action] in action_frequencies:
            action_frequencies[model.action_names[action]] += 1
        else:
            action_frequencies[model.action_names[action]] = 1
    
    return total_wins, total_reward, action_frequencies


def evaluate_agent(q_table, total_episodes=1000):
    win_rates = []
    average_rewards = []
    action_freqs_total = []
    
    for episode in tqdm(range(total_episodes)):  
        win_rate, avg_reward, action_freq = evaluation_game(q_table)
        win_rates.append(win_rate)
        average_rewards.append(avg_reward)
        action_freqs_total.append(action_freq)
        # print(f"Episode: {episode}, Win: {win_rate}, Reward: {avg_reward}")
        
    # print("Win-Rates: ", win_rates)
    plot_evaluation(action_freqs_total, win_rates, average_rewards)

#-------------------------Notes-----------------------------------------------*\
# 
#-----------------------------------------------------------------------------*\