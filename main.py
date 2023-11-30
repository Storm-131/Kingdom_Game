# ---------------------------------------------------------*\
# Title: Castle Game
# Author: TM
# ---------------------------------------------------------*/
#!/usr/bin/env python3

from src.model import train_q_learning, discretize_state, get_state_index, choose_action
from src.simulate import simulated_game
from src.Kingdom import Kingdom
from utils.helpers import analyze_q_table

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
            print(f"\n{player.name} successfully attacked! {opponent.name} \
                  lost {damage} castle strength and {damage} soldier(s).")
        else:
            print(f"\n{player.name}'s attack was repelled! {player.name} \
                  lost {damage} soldier(s).")
            
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
        print(f"\nRound {turn}")

        # Player 1's turn
        show_options_and_stats(kingdom1, kingdom2)
        choice = int(input("Choose your action: "))
        play_turn(kingdom1, kingdom2, choice)

        if kingdom2.castle_strength <= 0:
            print(f"\n{kingdom1.name} wins!")
            break
        
        # Player 2's turn
        show_options_and_stats(kingdom2, kingdom1)
        if opponent == "human":
            choice = int(input("Choose your action: "))
        else:
            choice = q_table_choice(kingdom1, kingdom2, q_table)
            
        play_turn(kingdom2, kingdom1, choice)
        
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
    
    ## Run nornal game

    game(opponent="machine")      # human / machine
    
    ## Run simulated game
    # simulated_game()

    # Train from scratch
    # trained_q_table = train_q_learning(total_episodes=100000)
    # result = analyze_q_table(trained_q_table)
    # print(result)
    
    # Train an already existing Q-Table
    # q_table_loaded = np.loadtxt("./out/q_table.csv", delimiter=",")
    # q_table_loaded = np.load("./out/q_table.npy")
    # trained_q_table = train_q_learning(q_table=q_table_loaded, total_episodes=10000, epsilon=0, alpha=0)
    
    

# -------------------------Notes-----------------------------------------------*\
#
# -----------------------------------------------------------------------------*\