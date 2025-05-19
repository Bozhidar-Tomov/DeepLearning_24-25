import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GRAY1 = (100, 100, 100)
GRAY2 = (150, 150, 150)

BLOCK_SIZE = 20
PADDING_SIZE = 4
MAX_OBSTACLES = 50
SPEED = 200

SCALE = 5
POSITIVE_REWARD = 12
NEGATIVE_REWARD = -10

class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.max_distance = np.linalg.norm(np.array([self.w, self.h]))
        self.epsilon = 1.0
        
        self.food = Point(0, 0)
        self.obstacles = []

        self.display = pygame.display.set_mode((self.w, self.h), pygame.NOFRAME)
        pygame.display.set_caption('Environment')
        
        self.clock = pygame.time.Clock()
        self.reset()
        
    def get_obstacle_count(self):
        max_limit = 55
        count = int((1 - self.epsilon) * max_limit)
        return min(count, max_limit)

    def reset(self):
        # init game state, snake is in the middle of the screen
        self.direction = Direction.RIGHT

        self.head = Point(self.w//2, self.h//2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self._place_food()
        self._place_obstacles()
        
        self.frame_iteration = 0
        self.score = 0

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake or self.food in self.obstacles:
            self._place_food()

    def _place_obstacles(self):
        self.obstacles = []
        attempts = 0
        while len(self.obstacles) < random.randint(0, self.get_obstacle_count()) and attempts < 1000:
            x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
            y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
            pt = Point(x, y)
            if pt != self.food and pt not in self.snake and pt not in self.obstacles:
                self.obstacles.append(pt)
            attempts += 1

    
    def play_step(self, action):
        self.frame_iteration += 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
                
        dist_old = np.linalg.norm(np.array([self.head.x, self.head.y]) - np.array([self.food.x, self.food.y]))

        self._move(action)
        self.snake.insert(0, self.head)
        
        dist_new = np.linalg.norm(np.array([self.head.x, self.head.y]) - np.array([self.food.x, self.food.y]))
        

        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            return NEGATIVE_REWARD , True , self.score # reward, is_game_over , self.score
        
        reward = 0
        dist_delta = (dist_old - dist_new) / self.max_distance
        reward += dist_delta * SCALE  # weight factor
        
        if self.head == self.food:
            self.score += 1
            reward = POSITIVE_REWARD
            self._place_food()
        else:
            self.snake.pop()
            
        self._update_ui()
        self.clock.tick(SPEED)

        return reward, False, self.score # reward, is_game_over , self.score


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself or obstacles
        elif pt in self.snake[1:] or pt in self.obstacles:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + PADDING_SIZE, pt.y + PADDING_SIZE,
                                                              BLOCK_SIZE - PADDING_SIZE*2,
                                                              BLOCK_SIZE - PADDING_SIZE*2))

        for pt in self.obstacles:
            pygame.draw.rect(self.display, GRAY1,
                             pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GRAY2, pygame.Rect(pt.x + PADDING_SIZE, pt.y + PADDING_SIZE,
                                                              BLOCK_SIZE - PADDING_SIZE*2,
                                                              BLOCK_SIZE - PADDING_SIZE*2))

        pygame.draw.rect(self.display, RED, pygame.Rect(
            self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN,
                      Direction.LEFT, Direction.UP]

        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx+1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx-1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
