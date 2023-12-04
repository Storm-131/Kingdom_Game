# ---------------------------------------------------------*\
# Title: Castle Game
# Author: TM
# ---------------------------------------------------------*/
#!/usr/bin/env python3

from src.Kingdom import Kingdom
from src.simulate import simulated_game
from utils.helpers import analyze_q_table
from src.model import train_q_learning, discretize_state, get_state_index, choose_action, evaluate_agent
import time

import numpy as np

#---------------------------------------------------------*/
# Classical Game (Against other human)
#---------------------------------------------------------*/
def play_turn(player, opponent, choice):

    if choice == 1:
        gathered = player.gather_resources()
        print(f"\n{player.name} gathered {gathered} resources.")
        
    elif choice == 2:
        trained = player.train_soldiers()
        if trained:
            print(f"\n{player.name} trained {trained} soldiers.")
        else:
            print("\nNot enough resources to train soldiers.")
            
    elif choice == 3:
        fortified = player.fortify_castle()
        if fortified:
            print(f"\n{player.name}'s castle has been fortified by {fortified}.")
        else:
            print("\nNot enough resources to fortify the castle.")

    elif choice == 4:
        success, damage = player.attack(opponent)
        if success:
            print(f"\n{player.name} successfully attacked! {opponent.name} " +
                  f"lost {damage} castle strength and {damage} soldier(s).")
        else:
            print(f"\n{player.name}'s attack was repelled! {player.name} " +
                f"lost {damage} soldier(s).")
            
def q_table_choice(kingdom1, kingdom2, q_table):
    state1 = discretize_state(kingdom1)
    state2 = discretize_state(kingdom2)
    state_index = get_state_index(state1, state2)
    return 1 + choose_action(q_table, state_index, epsilon=0)
    
def show_options_and_stats(player, opponent):
    """
    Display a friendly opening greeting with a cool ASCII header for the "Kingdom Game".
    This function shows the player's stats and available actions.
    """
    
    print("---------------------------------------------")
    print(f"\n{player.name}'s Turn:")
    print(f"Castle Strength: {player.castle_strength} (vs {opponent.castle_strength})")
    print(f"Resources: {player.resources} (vs {opponent.resources})")
    print(f"Soldiers: {player.soldiers} (vs {opponent.soldiers})")
    print("\nActions:")
    print("1. Gather Resources")
    print("2. Train Soldiers")
    print("3. Fortify Castle")
    print("4. Attack")
    

def game(limit=50, opponent="human"):
    kingdom1 = Kingdom("Kingdom A")
    kingdom2 = Kingdom("Kingdom B")
    q_table = np.load("./out/q_table.npy")
    turn = 1
    print (ascii_header)
    
    while True:
        print("\n---------------------------------------------")
        print(f"Round {turn}")

        # Player 1's turn
        show_options_and_stats(kingdom1, kingdom2)
        choice = int(input("\nChoose your action: "))
        play_turn(kingdom1, kingdom2, choice)

        if kingdom2.castle_strength <= 0:
            print(f"\n{kingdom1.name} wins!")
            break
        
        # Player 2's turn
        show_options_and_stats(kingdom2, kingdom1)
        if opponent == "human":
            choice = int(input("Choose your action: "))
            play_turn(kingdom2, kingdom1, choice)
        else:
            choice = q_table_choice(kingdom1, kingdom2, q_table)
            time.sleep(2)
            play_turn(kingdom2, kingdom1, choice)
            time.sleep(2)
                    
        if kingdom1.castle_strength <= 0:
            print(f"\n{kingdom2.name} wins!")
            break
        
        turn += 1
                
        if turn >= limit:
            print("\nGame ended in a draw!")
            break

# ASCII Art for "Kingdom Game"
ascii_header = """
   /\                                                        /\\
  |  |                                                      |  |
 /----\              Welcome to the Kingdom-Game           /----\\
[______]           ~ Where Brave Knights Tremble ~        [______]
 |    |         _____                        _____         |    |
 |[]  |        [     ]                      [     ]        |  []|
 |    |       [_______][ ][ ][ ][][ ][ ][ ][_______]       |    |
 |    [ ][ ][ ]|     |  ,----------------,  |     |[ ][ ][ ]    |
 |             |     |/'    ____..____    '\|     |             |
  \  []        |     |    /'    ||    '\    |     |        []  /
   |      []   |     |   |o     ||     o|   |     |  []       |
   |           |  _  |   |     _||_     |   |  _  |           |
   |   []      | (_) |   |    (_||_)    |   | (_) |       []  |
   |           |     |   |     (||)     |   |     |           |
   |           |     |   |      ||      |   |     |           |
 /''           |     |   |o     ||     o|   |     |           '\\
[_____________[_______]--'------''------'--[_______]_____________]                                 
    """
    
#---------------------------------------------------------*/
# Main-Loop
#---------------------------------------------------------*/

if __name__ == "__main__":
    
    choice = 5
    
    # 1) Nornal game (against human or machine)
    if choice == 1:  
        game(opponent="machine")  # human / machine
    
    # 2) Simulated game
    elif choice == 2: 
        simulated_game()  

    # 3) Train (Q-Table) from scratch
    elif choice == 3:
        trained_q_table = train_q_learning(total_episodes=100000)
        result = analyze_q_table(trained_q_table)
        print(result)
    
    # 4) Train with existing Q-Table
    elif choice == 4: 
        q_table_loaded = np.load("./out/q_table.npy")
        trained_q_table = train_q_learning(q_table=q_table_loaded, total_episodes=1000, epsilon=0, alpha=0)
        
    # 5) Evaluate Q-Table
    elif choice == 5:
        q_table_loaded = np.load("./out/q_table.npy")
        evaluate_agent(q_table_loaded, total_episodes=1000)

# -------------------------Notes-----------------------------------------------*\
#
# -----------------------------------------------------------------------------*\