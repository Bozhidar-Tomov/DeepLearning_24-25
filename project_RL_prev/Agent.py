import torch
import random
import numpy as np
from collections import deque
from Game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from Model import Linear_QNet, QTrainer
from Helper import plot

MAX_MEMORY = 150_000
BATCH_SIZE = 128
START_SAMPLING_SIZE = 1000
LEARNING_RATE = 0.005
STATE_SIZE = 11
HIDDEN_SIZE = 256
OUTPUT_SIZE = 3


class Agent():
    def __init__(self):
        self.number_of_games = 0
        self.epsilon = 1.0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(STATE_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        self.trainer = QTrainer(self.model, LEARNING_RATE, self.gamma)

    def get_state(self, game):
        head = game.snake[0]

        # Compute points relative to current direction
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Danger straight, right, left
        danger_straight = (
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d))
        )

        danger_right = (
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d))
        )

        danger_left = (
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d))
        )

        # Food direction relative to head
        food_left = game.food.x < head.x
        food_right = game.food.x > head.x
        food_up = game.food.y < head.y
        food_down = game.food.y > head.y

        state = [
            danger_straight,
            danger_right,
            danger_left,
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            food_left,
            food_right,
            food_up,
            food_down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_batch_experiences(self):

        if len(self.memory) > START_SAMPLING_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, done = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, done)

    def train_single_experience(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state):
        # Parameters
        epsilon_start = 1.0       
        epsilon_end = 0.005       
        epsilon_decay = 0.995   

        # Calculate current epsilon
        self.epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** self.number_of_games))

        final_move = [0, 0, 0]

        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[int(move)] = 1

        return final_move



def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        game.epsilon = agent.epsilon
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_single_experience(
            state_old, final_move, reward, state_new, done)

        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.number_of_games += 1
            agent.train_batch_experiences()

            if score > record:
                agent.model.save()
                
            record = max(record, score)
                
            plot_scores.append(score)
            total_score += score

            mean_score = total_score/agent.number_of_games
            plot_mean_scores.append(mean_score)
            
            if agent.number_of_games % 10 == 0:
                print(f"Game: {agent.number_of_games} | Score: {score} | Record: {record} | Epsilon: {agent.epsilon:.2f}")
                plot(plot_scores, plot_mean_scores)
                
        if agent.number_of_games > 1000:
            break


if __name__ == "__main__":
    train()
