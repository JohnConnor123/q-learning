import time
import random
import pickle
import numpy as np
import plotly.express as px
import pandas as pd
from tqdm import tqdm
import pygame

# для показа в PyQt5 вместо окна браузера
import sys, os
import plotly.offline
from PyQt5.QtCore import QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QApplication


class RaceEnv:
    width = 800
    height = 600
    agent_size = 20
        
    def __init__(self):
        pygame.init()
        self.window = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Race Simulation')
        self.track_boundary = pygame.Rect(50, 50, self.width - 100, self.height - 100)
        self.agent_pos = [self.track_boundary.x + self.agent_size, self.track_boundary.y + self.agent_size]
        self.agent_speed = 0
        self.agent_angle = 0

        # Препятствия (позиция и ширина/высота)
        self.obstacles = [
            pygame.Rect(300, 200, 50, 50),
            pygame.Rect(500, 400, 60, 60),
        ]
        
        # Точка финиша
        self.finish_line = pygame.Rect(self.width - 120, self.height - 120, 70, 70)

        self.actions = ['accelerate', 'brake', 'left', 'right']

    def reset(self):
        self.agent_pos = [self.track_boundary.x + self.agent_size, self.track_boundary.y + self.agent_size]
        self.agent_speed = 0
        self.agent_angle = 0
        state = self.get_state()
        return state

    def get_state(self):
        return (self.agent_pos[0], self.agent_pos[1], self.agent_speed, self.agent_angle) # x, y, speed, angle

    def step(self, action, training_step=0):
        reward = -2.5
        if action == 'accelerate':
            self.agent_speed += 1
        elif action == 'brake':
            self.agent_speed = max(0, self.agent_speed - 1)
        elif action == 'left':
            self.agent_angle -= 15
        elif action == 'right':
            self.agent_angle += 15

        rad_angle = np.deg2rad(self.agent_angle)
        self.agent_pos[0] += self.agent_speed * np.cos(rad_angle)
        self.agent_pos[1] += self.agent_speed * np.sin(rad_angle)

        agent_rect = pygame.Rect(self.agent_pos[0], self.agent_pos[1], self.agent_size, self.agent_size)

        # Если агент за границей трассы
        if not self.track_boundary.contains(agent_rect):
            reward = -150
            done = True

        elif any(obstacle.colliderect(agent_rect) for obstacle in self.obstacles):
            reward = -150
            done = True

        elif self.finish_line.colliderect(agent_rect):
            reward = 150  # Большая награда за пересечение финиша
            done = True
        else:
            reward += self.agent_speed - (training_step // 10) # Награда за движение вперед
            done = False

        next_state = self.get_state()
        return next_state, reward, done

    def render(self):
        self.window.fill((0, 0, 0))
        pygame.draw.rect(self.window, (255, 255, 255), self.track_boundary, 2)
        
        for obstacle in self.obstacles:
            pygame.draw.rect(self.window, (255, 0, 0), obstacle)
        
        pygame.draw.rect(self.window, (0, 0, 255), self.finish_line)

        agent_rect = pygame.Rect(self.agent_pos[0], self.agent_pos[1], self.agent_size, self.agent_size)
        pygame.draw.rect(self.window, (0, 255, 0), agent_rect)
        pygame.display.flip()

class QLearningAgent:
    def __init__(self, actions):
        # Q-table в моей реализации - это словарь, где ключом является кортеж, представляющий состояние, 
        # а значением - массив Q-значений для каждого действия
        self.q_table = {}
        self.actions = actions

        # чем больше, тем сильнее учитывается новый опыт по сравнению с предыдущим (т.е. агент обращает меньше внимания на накопленный опыт)
        self.learning_rate = 0.2

        # чем больше, тем сильнее учитывается будущее (доля награды за следующий шаг с отсутствием exploration)
        self.discount_factor = 0.9

        # P(exploration) = 0.1
        self.epsilon = 0.1 

    def get_state_key(self, state):
        x, y, speed, angle = state
        # машина ходит по 40 клеток, т.е. размерность Q-table уменьшается в 40х40=1600 раз
        state_key = (int(x // 40), int(y // 40), int(speed), angle)
        return state_key

    def choose_action(self, state):
        state_key = self.get_state_key(state)
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(self.actions)
        else:
            q_values = self.q_table.get(state_key, np.zeros(len(self.actions)))
            max_q = np.max(q_values)
            actions_with_max_q = [action for action, q in zip(self.actions, q_values) if q == max_q]
            action = random.choice(actions_with_max_q)
        return action

    def learn(self, state, action, reward, next_state):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        action_index = self.actions.index(action)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.actions))
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(len(self.actions))
        q_predict = self.q_table[state_key][action_index] # предсказанная награда
        q_target = reward + self.discount_factor * np.max(self.q_table[next_state_key]) # ожидаемая награда
        self.q_table[state_key][action_index] += self.learning_rate * (q_target - q_predict)

def train_agent(num_episodes=1000000):
    env = RaceEnv()
    agent = QLearningAgent(env.actions)
    num_episodes = num_episodes
    max_steps_per_episode = 200
    episode_rewards = []
    for episode in tqdm(range(num_episodes), desc="Training Progress"):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        while not done and step < max_steps_per_episode:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action, step)
            agent.learn(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            step += 1
        episode_rewards.append(total_reward)
        #print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")
    pygame.quit()

    save_q_table(agent.q_table)
    return episode_rewards

def save_q_table(q_table):
    with open('q_table.pkl', 'wb') as f:
        pickle.dump(q_table, f)

def load_trained_q_table():
    with open('q_table.pkl', 'rb') as f:
        q_table = pickle.load(f)
    return q_table

def run_agent():
    env = RaceEnv()
    agent = QLearningAgent(env.actions)
    agent.q_table = load_trained_q_table()
    state = env.reset()
    done = False
    clock = pygame.time.Clock()
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        state = next_state
        env.render()
        clock.tick(30) # 30 fps

        # если пользователь захотел выйти раньше времени
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
    pygame.quit()

def visualize_rewards(rewards_history):
    rewards_history = pd.DataFrame({'episode': range(1, len(rewards_history) + 1), 'reward': rewards_history})
    fig = px.line(
        rewards_history,
        x='episode', y='reward',
        template='plotly_dark',
        title="Reward history",
    )
    fig.update_layout(title_x=0.5)
    # fig.show() # для показа в браузере
    plotly.offline.plot(fig, filename='Report.html', auto_open=False)
    show_in_window() # для показа в PyQt5

def show_in_window():
    app = QApplication(sys.argv)
    web = QWebEngineView()
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Report.html"))
    web.load(QUrl.fromLocalFile(file_path))
    web.setWindowTitle("Report")
    web.resize(1500, 900)

    web.show()
    app.exec_()


if __name__ == "__main__":
    # обучение на 1e6 эпизодов занимает 10 минут
    is_training = False
    if is_training:
        num_episodes = int(1e6)
        rewards_history = train_agent(num_episodes)
        visualize_rewards(rewards_history)
    else:
        show_in_window()
        
    # чем быстрее, тем лучше прошло обучение
    n_runs = 10
    run_times = []
    for i in tqdm(range(n_runs), desc="Calculating average inference time"):
        start_time = time.time()
        run_agent()
        end_time = time.time()
        run_times.append(end_time - start_time)
    print(f"Time taken: {round(sum(run_times) / n_runs, 2)} seconds")
