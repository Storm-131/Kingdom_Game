#---------------------------------------------------------*\
# Title: Kingdom-Class
# Author: 
#---------------------------------------------------------*/
#!/usr/bin/env python3

import random

class Kingdom:
    def __init__(self, name, player="human"):
        self.name = name
        self.player = player
        self.castle_strength = 100
        self.resources = 50
        self.soldiers = 50

    def gather_resources(self):
        gathered = 10
        self.resources += gathered
        return gathered

    def train_soldiers(self):
        if self.resources >= 10:
            trained = 5
            self.resources -= 10
            self.soldiers += trained
            return trained
        else:
            return 0

    def fortify_castle(self):
        if self.resources >= 20:
            fortified = 10
            self.castle_strength += fortified
            self.resources -= 20
            return fortified
        else:
            return 0

    def attack(self, opponent):
        
        # Adjusting for the case where soldiers might be negative
        attack_strength = max(int(self.soldiers*0.333 + 3), 0) # + random.randint(-3, 3)
        defense_strength = max(int(opponent.soldiers*0.333), 0)

        # print(f"{attack_strength} vs {defense_strength}")

        # Determine the outcome of the battle
        if attack_strength > defense_strength:
            
            damage = attack_strength - defense_strength

            # Reduce castle strength and soldiers
            opponent.castle_strength -= damage
            opponent.castle_strength = max(0, opponent.castle_strength)  # Castle strength >= 0
            
            opponent.soldiers -= damage
            opponent.soldiers = max(0, opponent.soldiers)  # Soldiers >= 0
            
            return True, damage
        
        else:
            damage = defense_strength - attack_strength
            
            self.soldiers -= damage
            self.soldiers = max(0, self.soldiers)  # Ensure soldiers don't go below 0
            
            return False, damage

#-------------------------Notes-----------------------------------------------*\
# 
#-----------------------------------------------------------------------------*\
