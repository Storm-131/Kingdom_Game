# Documentation


## FAQ: How Q-Learning works

### 1. Initialization
- **Q-Table**: A table is initialized that stores a value (Q-value) for each combination of states and actions. These values represent the expected reward for performing an action in a given state.
- **Game Environment**: The environment of the game, i.e., the two kingdoms, is initialized. Each kingdom has its own castle strength, resources, and soldiers.

### 2. State Discretization
- Since the states can be continuous or very extensive, each state of the game (like the strength of the castle, the number of resources and soldiers) is converted into a discrete value that is easier to manage.

### 3. Action Selection
- In each turn, a kingdom chooses an action based on the current Q-table and the current state (e.g., attack, gather resources, train soldiers). This decision can be made either randomly or based on the highest Q-values ("ε-greedy").

### 4. Execute Action and Observe Outcome
- The kingdom executes the chosen action. The outcome of this action changes the state of the game (like changing the castle strength or the number of soldiers).
- For attacks, the outcome (success or failure) is also considered.

### 5. Reward Calculation
- Based on the changes in the state and the outcome of the action, a reward (or penalty) is calculated. This reward reflects how good the action was in relation to the game's goal.

### 6. Updating the Q-Table
- The Q-table is updated based on the received reward and the predicted future rewards. This step is central to the learning of the algorithm.

### 7. Repetition and Learning
- Steps 3 to 6 are repeated for a certain number of rounds or until a certain end of the game (like when a kingdom is destroyed).
- With each repetition, the program learns by adjusting the Q-values, determining which actions are best in which states.

### 8. End of the Game
- The game ends when a kingdom wins (e.g., destroys the opposing kingdom) or when a maximum number of rounds is reached.
- Optionally, a summary or analysis of the moves made and the learning progress can be performed.
 

The program uses the Q-Learning method to "learn" by repeated interaction with the game environment and observation of the outcomes of its actions. Over time, it adjusts its strategy to make better decisions for achieving the game's objective.

---

## Ideas for Improvements

Improving the performance of a Q-learning agent, especially in a complex or partially observable environment like the one you described, can be challenging. Here are several strategies you could try to enhance your agent's learning and increase its chances of winning:

1. **Reward Shaping**: Since your agent primarily receives a terminal reward, it might struggle to learn effective policies. Introducing intermediate rewards (or penalties) for certain actions or states could provide more frequent learning signals. For example, rewards for successfully gathering resources, training soldiers, or fortifying the castle could help guide the agent's learning process.

2. **Fine-Tuning Hyperparameters**: Adjusting hyperparameters such as the learning rate (alpha), discount factor (gamma), and exploration rate (epsilon) can significantly impact learning. Experiment with different values to find a balance between exploration and exploitation. You may also consider implementing more sophisticated epsilon decay strategies.

3. **State Representation**: Review how the states are represented and discretized. Ensure that the state representation captures all relevant information needed for the agent to make informed decisions. Sometimes, adding more details to the state or changing the way it's represented can improve the learning process.

4. **Action Space Review**: Ensure that the action space is appropriate for the problem. If there are actions that are rarely beneficial, consider removing them. Alternatively, if there are complex actions that could be broken down into simpler steps, that might also help.

5. **Learning from Past Experience (Experience Replay)**: Instead of learning from individual experiences as they occur, store these experiences in a replay buffer and randomly sample from this buffer to learn. This approach can help in breaking the correlation between consecutive learning steps and lead to more efficient learning.

6. **Increasing Episode Length or Training Duration**: If most games end in a draw, it might be beneficial to allow for longer episodes or more training episodes. This gives the agent more opportunity to explore different strategies and learn from them.

7. **Analyzing Agent's Decisions**: Periodically review the decisions made by your agent. Understanding why it makes certain choices can provide insights into what it has learned and what aspects of the environment it might be misunderstanding.

8. **Curriculum Learning**: Start by training your agent in simpler scenarios and gradually increase the complexity as it learns. This can help the agent to build up its understanding of the game incrementally.

9. **Debugging Learning Process**: Make sure your reward function, state transitions, and Q-table updates are working as expected. Sometimes bugs in these areas can significantly hinder learning.

10. **Alternative Learning Algorithms**: If Q-learning is not effective for your specific problem, consider exploring other reinforcement learning algorithms like SARSA, Deep Q-Networks (DQNs), or even policy gradient methods, which might be more suited to your environment.

Remember, reinforcement learning can be quite sensitive to small changes in the environment, reward structure, and hyperparameters. It often requires a lot of experimentation and tweaking to get right.


---

## Balancing the Game

Balancing a game like the one you've described, especially the attack mechanics, is a nuanced process that involves ensuring fairness, strategy, and fun for both players. Here are some suggestions to consider for balancing the `attack` method in your `Kingdom` class:

1. **Scaling Damage and Defense**: Consider how attack and defense strengths scale with the number of soldiers. The current formula (`soldiers * 0.333`) is a linear scale. You might want to experiment with different scaling factors to find a balance that feels right.

2. **Randomness Factor**: Incorporating a randomness factor (like `random.randint(-3, 3)`) can add unpredictability to battles, making them less deterministic. This could make the game more dynamic and interesting. However, too much randomness can also lead to frustration if players feel like their strategic decisions don't matter.

3. **Resource Utilization**: Currently, resources are used for training soldiers and fortifying the castle but not directly in attacking. Consider having attacks also cost resources, which would add a strategic layer to resource management.

4. **Soldier Losses in Defense**: When an attack fails, the attacking kingdom loses soldiers. You might also want to apply a similar mechanic for the defending side, even if they successfully defend. This can simulate the cost of battle.

5. **Balanced Attack and Defense Outcomes**: Ensure that the outcomes of attacks and defenses are balanced. For instance, if an attack is successful, the damage to the opponent should be significant enough to justify the risk. Conversely, a successful defense should also have tangible benefits.

6. **Limitations on Repeated Actions**: To prevent players from spamming a single action (like constantly attacking), you could introduce mechanics that incentivize a variety of strategies. For example, soldiers could become less effective if used for attacking multiple times in a row without rest or training.

7. **Winning Conditions**: Review how these mechanics influence the overall winning conditions of the game. Ensure that it’s possible to make a comeback or that one strategy doesn’t become overwhelmingly dominant.