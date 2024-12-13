import matplotlib.pyplot as plt
import pygame
import sys
import random


# Parameters
BOARD_ROWS = 10
BOARD_COLS = 10
CELL_SIZE = 50
WINDOW_SIZE = (BOARD_COLS * CELL_SIZE, BOARD_ROWS * CELL_SIZE)
TREASURE_POSITIONS = [(1, 2), (1, 4), (8, 2), (5, 8)]
WALLS = [
    ((4, 0), (4, 1)), ((4, 3), (4, 4)), ((0, 4), (4, 4)),
    ((0, 7), (4, 7)), ((2, 6), (2, 7)), ((6, 5), (6, 6)),
    ((7, 0), (7, 3)), ((6, 5), (10, 5)), ((5, 8), (10, 8))
]

# Colors
BACKGROUND_COLOR = (255, 255, 255)
WALL_COLOR = (0, 0, 0)
BORDER_COLOR = (200, 200, 200)
TREASURE_COLOR = (255, 223, 0)
AGENT_COLOR = (0, 128, 255)
COLLECTED_COLOR = (128, 255, 128)


def generate_wall_segments(walls):
    segments = []
    for start, end in walls:
        if start[0] == end[0]:
            row = start[0]
            for col in range(min(start[1], end[1]), max(start[1], end[1])):
                segments.append(((row, col), (row, col + 1)))
        elif start[1] == end[1]:
            col = start[1]
            for row in range(min(start[0], end[0]), max(start[0], end[0])):
                segments.append(((row, col), (row + 1, col)))
        else:
            raise ValueError("Walls must be horizontal or vertical.")
    return segments


WALL_SEGMENTS = generate_wall_segments(WALLS)


def random_start_position():
    row = random.randint(0, BOARD_ROWS - 1)
    col = random.randint(0, BOARD_COLS - 1)
    position = (row, col)
    return position


class State:
    def __init__(self):
        self.state = random_start_position()
        self.treasure_status = [0] * len(TREASURE_POSITIONS)
        self.isEnd = False

    def give_reward(self):
        # the agent is collecting a treasure
        if self.state in TREASURE_POSITIONS:
            idx = TREASURE_POSITIONS.index(self.state)
            if self.treasure_status[idx] == 0:
                # reward 10 for the treasure
                return 10
        # penalty 1 for moving
        return -1

    def is_end(self):
        self.isEnd = all(self.treasure_status)

    def collect_treasure(self):
        if self.state in TREASURE_POSITIONS:
            idx = TREASURE_POSITIONS.index(self.state)
            if self.treasure_status[idx] == 0:
                self.treasure_status[idx] = 1


def is_valid_move(current_position, next_position):
    current_row, current_col = current_position
    next_row, next_col = next_position
    if not (0 <= next_row < BOARD_ROWS and 0 <= next_col < BOARD_COLS):
        return False

    if next_row < current_row:
        if ((current_row, current_col), (current_row, current_col + 1)) in WALL_SEGMENTS:
            return False
    elif next_row > current_row:
        if ((next_row, current_col), (next_row, current_col + 1)) in WALL_SEGMENTS:
            return False
    elif next_col < current_col:
        if ((current_row, current_col), (current_row + 1, current_col)) in WALL_SEGMENTS:
            return False
    elif next_col > current_col:
        if ((current_row, next_col), (current_row + 1, next_col)) in WALL_SEGMENTS:
            return False

    return True


class Agent:
    def __init__(self, beta=0.6, alpha=0.95):
        self.actions = ["up", "down", "left", "right"]
        self.State = State()
        self.beta = beta  # learning rate
        self.alpha = alpha  # discount factor
        self.Q_values = {}
        self.isEnd = self.State.isEnd
        self.total_reward = 0

        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                self.Q_values[(row, col)] = {action: 0 for action in self.actions}

    def choose_action(self, epoch):
        n = random.uniform(0, 1)
        epsilon = max(0.1, 1.0 - epoch * 0.01)
        if n < epsilon:
            # exploration
            return random.choice(self.actions)
        else:
            # exploitation
            current_position = self.State.state
            return max(self.Q_values[current_position], key=self.Q_values[current_position].get)

    def take_action(self, action):
        current_position = self.State.state
        next_position = current_position

        if action == "up":
            next_position = (self.State.state[0] - 1, self.State.state[1])
        elif action == "down":
            next_position = (self.State.state[0] + 1, self.State.state[1])
        elif action == "left":
            next_position = (self.State.state[0], self.State.state[1] - 1)
        elif action == "right":
            next_position = (self.State.state[0], self.State.state[1] + 1)

        if is_valid_move(current_position, next_position):
            self.State.state = next_position

    def reset(self):
        self.State = State()
        self.isEnd = self.State.isEnd
        self.total_reward = 0

    def play_step(self, epoch):
        if self.State.isEnd:
            return True
        else:
            action = self.choose_action(epoch)
            self.take_action(action)

            reward = self.State.give_reward()
            self.total_reward += reward

            self.State.collect_treasure()
            next_state = self.State.state
            self.update_q_values(action, reward, next_state)
            self.State.is_end()
            return False

    def update_q_values(self, action, reward, next_state):
        current_position = self.State.state
        next_max_q = max(self.Q_values[next_state].values())
        current_q_value = self.Q_values[current_position][action]
        self.Q_values[current_position][action] = (current_q_value + self.beta *
                                                   (reward + self.alpha * next_max_q - current_q_value))

    def print_q_table(self):
        print("\nQ-Table:")
        for state, actions in self.Q_values.items():
            print(f"State {state}: {actions}")


class GridWorldDisplay:
    def __init__(self, window_size):
        pygame.init()
        self.screen = pygame.display.set_mode(window_size)

    def draw(self, agent):
        self.screen.fill(BACKGROUND_COLOR)

        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if (row, col) in TREASURE_POSITIONS:
                    idx = TREASURE_POSITIONS.index((row, col))
                    color = COLLECTED_COLOR if agent.State.treasure_status[idx] == 1 else TREASURE_COLOR
                    pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, BORDER_COLOR, rect, 1)

        for wall in WALL_SEGMENTS:
            (row1, col1), (row2, col2) = wall
            x1, y1 = col1 * CELL_SIZE, row1 * CELL_SIZE
            if row1 == row2:
                pygame.draw.line(self.screen, WALL_COLOR, (x1, y1), (x1 + CELL_SIZE, y1), 5)
            elif col1 == col2:
                pygame.draw.line(self.screen, WALL_COLOR, (x1, y1), (x1, y1 + CELL_SIZE), 5)

        agent_pos = agent.State.state
        agent_rect = pygame.Rect(agent_pos[1] * CELL_SIZE, agent_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(self.screen, AGENT_COLOR, agent_rect)

        pygame.display.flip()


def main():
    total_epochs = 3
    steps_per_epoch = 30
    max_moves_per_step = 400
    print_q_table = False

    agent = Agent()
    display = GridWorldDisplay(WINDOW_SIZE)
    rewards_per_step = []

    for epoch in range(1, total_epochs + 1):
        for step in range(steps_per_epoch):
            agent.reset()
            move_count = 0
            finished = False

            while move_count < max_moves_per_step and not finished:
                pygame.display.set_caption(f"Grid World. Epoch {epoch}, Step {step + 1}")
                display.draw(agent)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                finished = agent.play_step(epoch)
                move_count += 1
                pygame.time.delay(1)
            rewards_per_step.append(agent.total_reward)
        if print_q_table:
            agent.print_q_table()

    print("Training complete!")
    print("The Q-Table is: ")
    agent.print_q_table()

    plt.plot(range(len(rewards_per_step)), rewards_per_step, marker='o')
    plt.xlabel("Steps and Epochs")
    plt.ylabel("Reward")
    plt.title("Final Reward")
    plt.grid()
    plt.show()

    pygame.quit()


if __name__ == "__main__":
    main()
