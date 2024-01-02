# ---------------------------------------------------------*\
# Title: Kingdom Game
# Author: TM
# ---------------------------------------------------------*/
#!/usr/bin/env python3

import numpy as np
from models.Q_Learning_model import QLearningAgent
from src.simulate import simulated_game
from utils.helpers import analyze_q_table
from src.game import game
from src.eval import evaluate_agent
# from models.model import train_q_learning, evaluate_agent

#---------------------------------------------------------*/
# Main-Loop
#---------------------------------------------------------*/

if __name__ == "__main__":
    
    model = QLearningAgent(total_episodes=10000)
    
    choice = 3
    
    # 1) Nornal game (against human or machine)
    if choice == 1:  
        game(opponent="machine")  # human / machine1
    
    # 2) Simulated game
    elif choice == 2: 
        simulated_game()  

    # 3) Train (Q-Table) from scratch
    elif choice == 3:
        trained_q_table = model.train_q_learning()
        result = analyze_q_table(trained_q_table)
        print(result)
    
    # 4) Train with existing Q-Table
    elif choice == 4: 
        q_table_loaded = np.load("./out/q_table.npy")
        trained_q_table = model.train_q_learning(q_table=q_table_loaded, total_episodes=10000, epsilon=0, alpha=0)
        
    # 5) Evaluate Q-Table
    elif choice == 5:
        q_table_loaded = np.load("./out/q_table.npy")
        evaluate_agent(q_table_loaded, total_episodes=1000)

# -------------------------Notes-----------------------------------------------*\
#
# -----------------------------------------------------------------------------*\