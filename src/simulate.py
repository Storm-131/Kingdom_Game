#---------------------------------------------------------*\
# Title: Simulate a game
# Author: 
#---------------------------------------------------------*/
#!/usr/bin/env python3
from src.Kingdom import Kingdom
import random

def simulated_play_turn(player, opponent):
    print(f"\n{player.name}'s Turn:")
    print(f"Castle Strength: {player.castle_strength}")
    print(f"Resources: {player.resources}")
    print(f"Soldiers: {player.soldiers}")

    choice = random.randint(1, 4)

    if choice == 1:
        success, damage = player.attack(opponent)
        if success:
            print(
                f"\n{player.name} successfully attacked! {opponent.name} lost {damage} castle strength and {damage//2} soldiers."
            )
        else:
            print(
                f"\n{player.name}'s attack was repelled! {player.name} lost {damage//2} soldiers."
            )
    elif choice == 2:
        gathered = player.gather_resources()
        print(f"\n{player.name} gathered {gathered} resources.")
    elif choice == 3:
        trained = player.train_soldiers()
        if trained:
            print(f"\n{player.name} trained {trained} soldiers.")
        else:
            print("\nNot enough resources to train soldiers.")
    elif choice == 4:
        fortified = player.fortify_castle()
        if fortified:
            print(f"\n{player.name}'s castle has been fortified by {fortified}.")
        else:
            print("\nNot enough resources to fortify the castle.")


def simulated_game():
    kingdom1 = Kingdom("Kingdom A")
    kingdom2 = Kingdom("Kingdom B")

    turn = 1
    while True:
        print(f"\nTurn {turn}")
        simulated_play_turn(kingdom1, kingdom2)
        if kingdom1.soldiers <= 0 or kingdom1.castle_strength <= 0:
            print(f"\n{kingdom2.name} wins!")
            break

        simulated_play_turn(kingdom2, kingdom1)
        if kingdom2.soldiers <= 0 or kingdom2.castle_strength <= 0:
            print(f"\n{kingdom1.name} wins!")
            break

        turn += 1
        if turn > 50:  # Adding a maximum turn limit to prevent potential endless games
            print("\nGame ended in a draw after 50 turns!")
            break


#-------------------------Notes-----------------------------------------------*\
# 
#-----------------------------------------------------------------------------*\