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
