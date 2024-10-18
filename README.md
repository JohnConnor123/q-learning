## Basic definitions
Reinforcement Learning (RL) is a machine learning method where an agent learns to interact with an environment by choosing actions to maximize rewards. The key elements are:
1. **Agent** — the system making decisions.
2. **Environment** — the world the agent interacts with.
3. **State** — a description of the agent's current position in the environment.
4. **Action** — the choice the agent makes, which influences the transition to the next state.
5. **Reward** — a numerical score the agent receives after performing an action.

The connection here is as follows: the agent perceives the state of the environment, selects an action, receives a reward, and updates its behavior to maximize the total future reward.

There is also such an important concept as **policy**: is the strategy the agent uses to choose actions in each state. It defines what action the agent will take based on the current state.

## Example
In a game, an agent (e.g., a robot) decides how to move through a maze. The policy tells the agent how to choose directions at each state of the maze to reach the goal faster and earn maximum reward (e.g., minimizing the distance traveled).

## Analysis of results:
The agent implemented in main.py performs well - it reaches the finish line in more than 90% of cases, avoiding obstacles. Agent

## Ideas for improvement
We can slow down ε-greedy exploration over time. Instead of fixing the probability of a random action (ε), we can start with a higher exploration probability and gradually decrease it as it learns, so that the agent chooses actions with the highest Q-value more often.

## Factors affecting learning
1. **State size and structure** — the size and representation of the state affect the ability of the agent to learn. It is important to discretize the state space correctly so that the agent can effectively explore parameters such as speed and turning angle without losing information.
2. **Reward function** — rewards should be balanced: frequent enough to incentivize useful actions, but not too frequent so as not to reduce the significance of progress. It is useful to break the main task into subtasks and provide rewards for completing them to keep the agent motivated and reward it for reducing the distance to the finish line.
