import numpy as np
import random
import os
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import time


class DQNetwork(nn.Module):
    def __init__(self, state_size=24, action_size=24):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class NineMensMorrisEnv:
    def __init__(self):
        
        self.board = [0] * 24
        self.phase = 'placement'  
        self.remaining_pieces = {1: 9, 2: 9}
        self.pieces_on_board = {1: 0, 2: 0}
        self.mills = []
        
        self.all_mills = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23],  
            [0, 9, 21], [3, 10, 18], [6, 11, 15], [1, 4, 7], [16, 19, 22], [8, 12, 17], [5, 13, 20], [2, 14, 23]  
        ]
        
        self.central_positions = [4, 10, 13, 16, 19] 

    def reset(self):
        self.board = [0] * 24
        self.phase = 'placement'
        self.remaining_pieces = {1: 9, 2: 9}
        self.pieces_on_board = {1: 0, 2: 0}
        self.mills = []
        return self.get_state()

    def get_state(self):
        return np.array(self.board) / 2.0

    def get_valid_actions(self, player):
        actions = []
        if self.phase == 'placement':
            for pos in range(24):
                if self.board[pos] == 0:
                    actions.append(pos)
        elif self.phase in ['movement', 'flying']:
            for pos in range(24):
                if self.board[pos] == player:
                    if self.phase == 'flying' and self.pieces_on_board[player] == 3:
                        for new_pos in range(24):
                            if self.board[new_pos] == 0:
                                actions.append((pos, new_pos))
                    else:
                        for new_pos in self.get_adjacent(pos):
                            if self.board[new_pos] == 0:
                                actions.append((pos, new_pos))
        return actions

    def get_adjacent(self, position):
        adjacency = {
            0: [1, 9], 1: [0, 2, 4], 2: [1, 14], 3: [4, 10], 4: [1, 3, 5, 7], 5: [4, 13], 6: [7, 11],
            7: [4, 6, 8], 8: [7, 12], 9: [0, 10, 21], 10: [3, 9, 11, 18], 11: [6, 10, 15], 12: [8, 13, 17],
            13: [5, 12, 14, 20], 14: [2, 13, 23], 15: [11, 16], 16: [15, 17, 19], 17: [12, 16],
            18: [10, 19], 19: [16, 18, 20, 22], 20: [13, 19], 21: [9, 22], 22: [19, 21, 23], 23: [14, 22]
        }
        return adjacency[position]

    def check_mill(self, position, player):
        mills = self.all_mills
        new_mills = []
        for mill in mills:
            if position in mill and all(self.board[pos] == player for pos in mill):
                if mill not in self.mills:
                    new_mills.append(mill)
        if new_mills:
            self.mills.extend(new_mills)
            return True
        return False

    def can_remove(self, opponent):
        all_in_mills = True
        for pos in range(24):
            if self.board[pos] == opponent:
                in_mill = False
                for mill in self.mills:
                    if pos in mill and all(self.board[m] == opponent for m in mill):
                        in_mill = True
                        break
                if not in_mill:
                    all_in_mills = False
                    return True
        return all_in_mills and self.pieces_on_board[opponent] > 0

    def place_piece(self, position, player):
        if self.board[position] == 0 and self.remaining_pieces[player] > 0:
            self.board[position] = player
            self.remaining_pieces[player] -= 1
            self.pieces_on_board[player] += 1
            formed_mill = self.check_mill(position, player)
            if self.remaining_pieces[1] == 0 and self.remaining_pieces[2] == 0:
                self.phase = 'movement'
            return formed_mill
        return False

    def move_piece(self, from_pos, to_pos, player):
        if self.board[from_pos] == player and self.board[to_pos] == 0:
            if self.phase == 'flying' and self.pieces_on_board[player] == 3:
                valid_move = True
            else:
                valid_move = to_pos in self.get_adjacent(from_pos)
            if valid_move:
                self.board[from_pos] = 0
                self.board[to_pos] = player
                
                self.update_mills()  
                return self.check_mill(to_pos, player)
        return False

    def remove_piece(self, position, opponent):
        if self.board[position] == opponent:
            in_mill = False
            for mill in self.mills:
                if position in mill and all(self.board[pos] == opponent for pos in mill):
                    in_mill = True
                    break
            if not in_mill or self.all_pieces_in_mills(opponent):
                self.board[position] = 0
                self.pieces_on_board[opponent] -= 1
                self.update_mills()
                if self.pieces_on_board[opponent] == 3 and self.phase == 'movement':
                    self.phase = 'flying'
                return True
        return False

    def all_pieces_in_mills(self, player):
        for pos in range(24):
            if self.board[pos] == player:
                in_mill = False
                for mill in self.mills:
                    if pos in mill and all(self.board[m] == player for m in mill):
                        in_mill = True
                        break
                if not in_mill:
                    return False
        return True

    def update_mills(self):
        valid_mills = []
        for mill in self.mills:
            player = self.board[mill[0]]
            if player != 0 and all(self.board[pos] == player for pos in mill):
                valid_mills.append(mill)
        self.mills = valid_mills

    def check_winner(self):
        if self.pieces_on_board[1] < 3 and self.remaining_pieces[1] == 0:
            return 2
        if self.pieces_on_board[2] < 3 and self.remaining_pieces[2] == 0:
            return 1
        if self.phase in ['movement', 'flying']:
            if self.pieces_on_board[1] > 0 and not self.get_valid_actions(1):
                return 2
            if self.pieces_on_board[2] > 0 and not self.get_valid_actions(2):
                return 1
        return 0

    def get_potential_mills(self, board, player):
        """Identify mill lines with exactly two player pieces and one empty spot."""
        potential_mills = []
        for mill in self.all_mills:
            pieces = [board[pos] for pos in mill]
            if pieces.count(player) == 2 and pieces.count(0) == 1:
                potential_mills.append(mill)
        return potential_mills

    def step(self, action, player):
        opponent = 3 - player
        
        prev_board = self.board.copy()

        
        if self.phase == 'placement':
            formed_mill = self.place_piece(action, player)
        else:
            from_pos, to_pos = action
            formed_mill = self.move_piece(from_pos, to_pos, player)

        reward = 0
        done = False

        
        if formed_mill and self.can_remove(opponent):
            for pos in range(24):
                if self.board[pos] == opponent:
                    in_mill = False
                    for mill in self.mills:
                        if pos in mill and all(self.board[m] == opponent for m in mill):
                            in_mill = True
                            break
                    if not in_mill:
                        self.remove_piece(pos, opponent)
                        reward += 2.0
                        break
            else:
                for pos in range(24):
                    if self.board[pos] == opponent:
                        self.remove_piece(pos, opponent)
                        reward += 2.0
                        break

        
        winner = self.check_winner()
        if winner == player:
            reward += 10.0
            done = True
        elif winner == opponent:
            reward += -10.0
            done = True

        
        if formed_mill:
            reward += 1.0  

        
        if self.phase == 'placement':
            if action in self.central_positions:
                reward += 0.05
        else:
            
            to_pos=action
            if to_pos in self.central_positions:
                reward += 0.05

        
        prev_potential_player = self.get_potential_mills(prev_board, player)
        current_potential_player = self.get_potential_mills(self.board, player)
        new_potential_mills = set(tuple(sorted(m)) for m in current_potential_player) - set(tuple(sorted(m)) for m in prev_potential_player)
        reward += 0.1 * len(new_potential_mills)

        
        prev_potential_opponent = self.get_potential_mills(prev_board, opponent)
        current_potential_opponent = self.get_potential_mills(self.board, opponent)
        blocked_mills = set(tuple(sorted(m)) for m in prev_potential_opponent) - set(tuple(sorted(m)) for m in current_potential_opponent)
        reward += 0.5 * len(blocked_mills)

        
        piece_diff = self.pieces_on_board[player] - self.pieces_on_board[opponent]
        reward += 0.2 * piece_diff

        
        reward += -0.05

        return self.get_state(), reward, done




class DQNAgent:
    def __init__(self, state_size=24, action_size=24, device="cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        self.model = DQNetwork(state_size, action_size).to(device)
        self.target_model = DQNetwork(state_size, action_size).to(device)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        self.replay_buffer = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.batch_size = 32

        
        self.total_episodes = 0

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def act(self, state, valid_actions, epsilon_override=None):
        
        epsilon = epsilon_override if epsilon_override is not None else self.epsilon

        if np.random.rand() <= epsilon:
            return random.choice(valid_actions)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy()[0]

        
        if isinstance(valid_actions[0], (int, np.int64)):
            q_valid = [-float('inf')] * self.action_size
            for a in valid_actions:
                q_valid[a] = q_values[a]
            return np.argmax(q_valid)
        
        else:
            best_action = valid_actions[0]
            best_q = -float('inf')
            for action in valid_actions:
                from_pos, to_pos = action
                
                action_idx = (from_pos + to_pos) % self.action_size
                if q_values[action_idx] > best_q:
                    best_q = q_values[action_idx]
                    best_action = action
            return best_action

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        minibatch = random.sample(self.replay_buffer, self.batch_size)

        states = np.array([x[0] for x in minibatch])

        
        actions_raw = [x[1] for x in minibatch]
        actions = []
        for a in actions_raw:
            if isinstance(a, (int, np.int64)):
                actions.append(a)
            else:
                
                from_pos, to_pos = a
                actions.append((from_pos + to_pos) % self.action_size)

        
        action_indices = []
        for a in actions:
            if isinstance(a, (int, np.int64)):
                action_indices.append(a)
            elif isinstance(a, tuple):
                action_indices.append(a[0])  
            else:
                
                action_indices.append(0)

        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])

        
        states_tensor = torch.FloatTensor(states).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        actions_tensor = torch.LongTensor(action_indices).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        
        current_q = self.model(states_tensor).gather(1, actions_tensor)

        
        with torch.no_grad():
            next_q = torch.zeros(self.batch_size, device=self.device)
            next_max_q = self.target_model(next_states_tensor).max(1)[0]
            next_q = rewards_tensor + (1 - dones_tensor) * self.gamma * next_max_q

        
        target_q = next_q.unsqueeze(1)

        
        loss = self.criterion(current_q, target_q)

        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        
        replay_list = list(self.replay_buffer)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'replay_buffer': replay_list,
            'total_episodes': self.total_episodes
        }
        torch.save(checkpoint, filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        if os.path.exists(filename):
            try:
                
                try:
                    checkpoint = torch.load(filename, map_location=self.device, weights_only=False)
                except TypeError:
                    
                    checkpoint = torch.load(filename, map_location=self.device)

                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint['epsilon']

                
                if 'replay_buffer' in checkpoint:
                    self.replay_buffer = deque(maxlen=2000)
                    for item in checkpoint['replay_buffer']:
                        self.replay_buffer.append(item)

                
                if 'total_episodes' in checkpoint:
                    self.total_episodes = checkpoint['total_episodes']

                print(f"Model loaded from {filename}")
                print(f"Resumed from episode {self.total_episodes} with epsilon {self.epsilon:.4f}")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Starting with a fresh model state.")
                return False
        return False

def select_action(env, agent, player, difficulty):
    valid_actions = env.get_valid_actions(player)

    
    if difficulty == "easy":
        
        if np.random.random() < 0.7:
            return random.choice(valid_actions)

        
        for action in valid_actions:
            
            test_env = NineMensMorrisEnv()
            test_env.board = env.board.copy()
            test_env.phase = env.phase
            test_env.remaining_pieces = env.remaining_pieces.copy()
            test_env.pieces_on_board = env.pieces_on_board.copy()
            test_env.mills = [mill.copy() for mill in env.mills]

            
            if env.phase == 'placement':
                if test_env.place_piece(action, player):
                    return action
            else:
                from_pos, to_pos = action
                if test_env.move_piece(from_pos, to_pos, player):
                    return action

        
        return random.choice(valid_actions)

    elif difficulty == "moderate":
        
        if np.random.random() < 0.3:
            return random.choice(valid_actions)

        
        for action in valid_actions:
            
            test_env = NineMensMorrisEnv()
            test_env.board = env.board.copy()
            test_env.phase = env.phase
            test_env.remaining_pieces = env.remaining_pieces.copy()
            test_env.pieces_on_board = env.pieces_on_board.copy()
            test_env.mills = [mill.copy() for mill in env.mills]

            
            if env.phase == 'placement':
                if test_env.place_piece(action, player):
                    return action
            else:
                from_pos, to_pos = action
                if test_env.move_piece(from_pos, to_pos, player):
                    return action

        
        opponent = 3 - player
        opponent_actions = env.get_valid_actions(opponent)

        for opp_action in opponent_actions:
            
            test_env = NineMensMorrisEnv()
            test_env.board = env.board.copy()
            test_env.phase = env.phase
            test_env.remaining_pieces = env.remaining_pieces.copy()
            test_env.pieces_on_board = env.pieces_on_board.copy()
            test_env.mills = [mill.copy() for mill in env.mills]

            
            will_form_mill = False
            if test_env.phase == 'placement':
                will_form_mill = test_env.place_piece(opp_action, opponent)
            else:
                from_pos, to_pos = opp_action
                will_form_mill = test_env.move_piece(from_pos, to_pos, opponent)

            
            if will_form_mill:
                for action in valid_actions:
                    
                    block_test_env = NineMensMorrisEnv()
                    block_test_env.board = env.board.copy()
                    block_test_env.phase = env.phase
                    block_test_env.remaining_pieces = env.remaining_pieces.copy()
                    block_test_env.pieces_on_board = env.pieces_on_board.copy()
                    block_test_env.mills = [mill.copy() for mill in env.mills]

                    
                    if env.phase == 'placement':
                        block_test_env.place_piece(action, player)
                    else:
                        from_pos, to_pos = action
                        block_test_env.move_piece(from_pos, to_pos, player)

                    
                    if env.phase == 'placement':
                        
                        if block_test_env.board[opp_action] != 0:
                            return action  
                    else:
                        
                        from_pos, to_pos = opp_action
                        if block_test_env.board[to_pos] != 0:
                            return action  

        
        state = env.get_state()
        return agent.act(state, valid_actions, epsilon_override=0.2)

    else:  
        

        
        for action in valid_actions:
            
            test_env = NineMensMorrisEnv()
            test_env.board = env.board.copy()
            test_env.phase = env.phase
            test_env.remaining_pieces = env.remaining_pieces.copy()
            test_env.pieces_on_board = env.pieces_on_board.copy()
            test_env.mills = [mill.copy() for mill in env.mills]

            
            if env.phase == 'placement':
                if test_env.place_piece(action, player):
                    return action
            else:
                from_pos, to_pos = action
                if test_env.move_piece(from_pos, to_pos, player):
                    return action

        
        opponent = 3 - player
        opponent_actions = env.get_valid_actions(opponent)

        for opp_action in opponent_actions:
            
            test_env = NineMensMorrisEnv()
            test_env.board = env.board.copy()
            test_env.phase = env.phase
            test_env.remaining_pieces = env.remaining_pieces.copy()
            test_env.pieces_on_board = env.pieces_on_board.copy()
            test_env.mills = [mill.copy() for mill in env.mills]

            
            will_form_mill = False
            if test_env.phase == 'placement':
                will_form_mill = test_env.place_piece(opp_action, opponent)
            else:
                from_pos, to_pos = opp_action
                will_form_mill = test_env.move_piece(from_pos, to_pos, opponent)

            
            if will_form_mill:
                for action in valid_actions:
                    
                    block_test_env = NineMensMorrisEnv()
                    block_test_env.board = env.board.copy()
                    block_test_env.phase = env.phase
                    block_test_env.remaining_pieces = env.remaining_pieces.copy()
                    block_test_env.pieces_on_board = env.pieces_on_board.copy()
                    block_test_env.mills = [mill.copy() for mill in env.mills]

                    
                    if env.phase == 'placement':
                        block_test_env.place_piece(action, player)
                    else:
                        from_pos, to_pos = action
                        block_test_env.move_piece(from_pos, to_pos, player)

                    
                    if env.phase == 'placement':
                        
                        if block_test_env.board[opp_action] != 0:
                            return action  
                    else:
                        
                        from_pos, to_pos = opp_action
                        if block_test_env.board[to_pos] != 0:
                            return action  

        
        for action in valid_actions:
            
            test_env = NineMensMorrisEnv()
            test_env.board = env.board.copy()
            test_env.phase = env.phase
            test_env.remaining_pieces = env.remaining_pieces.copy()
            test_env.pieces_on_board = env.pieces_on_board.copy()
            test_env.mills = [mill.copy() for mill in env.mills]

            
            if env.phase == 'placement':
                test_env.place_piece(action, player)
                pos = action  
            else:
                from_pos, to_pos = action
                test_env.move_piece(from_pos, to_pos, player)
                pos = to_pos  

            
            mills = [
                
                [0, 1, 2], [3, 4, 5], [6, 7, 8],
                [9, 10, 11], [12, 13, 14], [15, 16, 17],
                [18, 19, 20], [21, 22, 23],
                
                [0, 9, 21], [3, 10, 18], [6, 11, 15],
                [1, 4, 7], [16, 19, 22], [8, 12, 17],
                [5, 13, 20], [2, 14, 23]
            ]

            
            for mill in mills:
                if pos in mill:
                    pieces_count = sum(1 for m_pos in mill if test_env.board[m_pos] == player)
                    empty_count = sum(1 for m_pos in mill if test_env.board[m_pos] == 0)

                    if pieces_count == 2 and empty_count == 1:
                        
                        return action

        
        state = env.get_state()
        return agent.act(state, valid_actions, epsilon_override=0.05)

def train_agent(episodes=1000, model_filename="nine_mens_morris_model.pth", resume=False):
    env = NineMensMorrisEnv()

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    
    agent = DQNAgent(device=device)

    
    if resume and os.path.exists(model_filename):
        agent.load(model_filename)
        start_episode = agent.total_episodes
    else:
        start_episode = 0

    for e in range(start_episode, start_episode + episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            
            valid_actions = env.get_valid_actions(2)
            if not valid_actions:
                break

            action = agent.act(state, valid_actions)
            next_state, reward, done = env.step(action, 2)
            agent.remember(state, action, reward, next_state, done)
            total_reward += reward

            
            if not done:
                valid_actions_opp = env.get_valid_actions(1)
                if valid_actions_opp:
                    
                    opp_action = random.choice(valid_actions_opp)
                    next_state_opp, _, done_opp = env.step(opp_action, 1)

                    
                    if env.check_winner() == 1:
                        reward = -1
                        done = True
                        agent.remember(next_state, opp_action, reward, next_state_opp, done)
                        total_reward += reward
                    else:
                        done = done_opp
                    state = next_state_opp
                else:
                    
                    done = True
                    reward = 1  
                    total_reward += reward

            agent.train()

        if e % 10 == 0:
            agent.update_target_model()

        if e % 100 == 0:
            print(f"Episode: {e+1}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

        
        if (e + 1) % 100 == 0:
            agent.total_episodes = e + 1
            agent.save(model_filename)

    
    agent.total_episodes = start_episode + episodes
    agent.save(model_filename)

    return agent

def print_board(board, highlight=None):
    """
    Beautiful board visualization for Nine Men's Morris with perfect alignment.

    Parameters:
    - board: The game board state (List or Tuple representing positions).
    - highlight: Optional position to highlight (useful for showing last move).
    """
    
    if os.name == 'posix':  
        os.system('clear')
    else:  
        os.system('cls')

    
    BOLD = "\033[1m"
    RESET = "\033[0m"
    BLUE = "\033[34m"
    RED = "\033[31m"
    GREEN = "\033[32m"

    
    symbols = {
        0: f" {BLUE}\u25CF{RESET} ",  
        1: f" {RED}X{RESET} ",   
        2: f" {GREEN}O{RESET} "  
    }

    
    highlight_symbols = {
        0: f"{BLUE}[\u25CF]{RESET}",
        1: f"{RED}[X]{RESET}",
        2: f"{GREEN}[O]{RESET}"
    }

    
    def sym(pos):
        if highlight is not None and pos == highlight:
            return highlight_symbols[board[pos]]
        else:
            return symbols[board[pos]]

    
    print("\n" + "╔" + "═" * 38 + "╗")
    print("║" + f"{BOLD}      NINE MEN'S MORRIS GAME      {RESET}" + "║")
    print("╚" + "═" * 38 + "╝\n")

    
    board_layout = (f"""
    {sym(0)}----------{sym(1)}----------{sym(2)}
    |          |                      |
    |  {sym(3)}-------{sym(4)}-------{sym(5)}  |
    |  |       |       |  |
    |  |  {sym(6)}----{sym(7)}----{sym(8)}  |  |
    |  |  |         |  |  |
  {sym(9)}-{sym(10)}-{sym(11)}            {sym(12)}-{sym(13)}-{sym(14)}
    |  |  |         |  |  |
    |  |  {sym(15)}----{sym(16)}----{sym(17)}  |  |
    |  |       |       |  |
    |  {sym(18)}-------{sym(19)}-------{sym(20)}  |
    |          |          |
    {sym(21)}----------{sym(22)}----------{sym(23)}
    """)

    print(board_layout)

    
    print("\n╔" + "═" * 38 + "╗")
    print(f"║ {BOLD}Legend: {RESET}                              ║")
    print(f"║ {RED}X{RESET} - Player 1    {GREEN}O{RESET} - Player 2    {BLUE}\u25CF{RESET} - Empty ║")
    print(f"║ [Symbol] - Last Move                 ║")
    print("╚" + "═" * 38 + "╝\n")

def play_game_vs_ai(agent, difficulty="moderate"):
    """
    Play a game of Nine Men's Morris against the AI

    Parameters:
    - agent: The trained DQN agent
    - difficulty: AI difficulty level ("easy", "moderate", "difficult")
    """
    env = NineMensMorrisEnv()
    state = env.reset()

    
    print("\nWould you like to go first? (y/n)")
    choice = input().lower().strip()
    player_first = choice == 'y'

    human_player = 1 if player_first else 2
    ai_player = 3 - human_player  

    current_player = 1  
    done = False
    last_move = None

    print(f"\nYou are playing as {'X' if human_player == 1 else 'O'}")
    print(f"AI difficulty: {difficulty}")
    print("\nGame starts!\n")

    while not done:
        print_board(env.board, last_move)

        
        print(f"\nPhase: {env.phase.capitalize()}")
        print(f"Player 1 (X): {env.remaining_pieces[1]} pieces to place, {env.pieces_on_board[1]} on board")
        print(f"Player 2 (O): {env.remaining_pieces[2]} pieces to place, {env.pieces_on_board[2]} on board")

        
        if current_player == human_player:
            print("\nYour turn!")

            valid_actions = env.get_valid_actions(human_player)
            if not valid_actions:
                print("You have no valid moves. You lose!")
                break

            if env.phase == 'placement':
                
                while True:
                    try:
                        print("Enter position to place your piece (0-23):")
                        pos = int(input())
                        if pos in valid_actions:
                            last_move = pos
                            formed_mill = env.place_piece(pos, human_player)
                            break
                        else:
                            print("Invalid position. Try again.")
                    except ValueError:
                        print("Please enter a valid number.")

            else:  
                
                while True:
                    try:
                        print("Enter position of piece to move (0-23):")
                        from_pos = int(input())

                        
                        valid_from = any(action[0] == from_pos for action in valid_actions)

                        if valid_from:
                            
                            valid_destinations = [action[1] for action in valid_actions if action[0] == from_pos]
                            print(f"Valid destinations: {valid_destinations}")

                            
                            print("Enter destination position:")
                            to_pos = int(input())

                            if to_pos in valid_destinations:
                                last_move = to_pos
                                formed_mill = env.move_piece(from_pos, to_pos, human_player)
                                break
                            else:
                                print("Invalid destination. Try again.")
                        else:
                            print("You cannot move that piece. Try again.")
                    except ValueError:
                        print("Please enter a valid number.")

            
            if formed_mill and env.can_remove(ai_player):
                print_board(env.board, last_move)
                print("\nYou formed a mill! Remove an opponent piece.")

                while True:
                    try:
                        print("Enter position of opponent piece to remove (0-23):")
                        pos = int(input())

                        
                        if env.board[pos] == ai_player:
                            in_mill = False
                            for mill in env.mills:
                                if pos in mill and all(env.board[m] == ai_player for m in mill):
                                    in_mill = True
                                    break

                            
                            if in_mill and not env.all_pieces_in_mills(ai_player):
                                print("Cannot remove a piece that's part of a mill unless no other pieces are available.")
                            else:
                                env.remove_piece(pos, ai_player)
                                break
                        else:
                            print("Invalid position. Select an opponent piece.")
                    except ValueError:
                        print("Please enter a valid number.")

        
        else:
            print(f"\nAI ({difficulty}) is thinking...")
            time.sleep(1)  

            valid_actions = env.get_valid_actions(ai_player)
            if not valid_actions:
                print("AI has no valid moves. You win!")
                break

            action = select_action(env, agent, ai_player, difficulty)

            if env.phase == 'placement':
                print(f"AI places a piece at position {action}")
                last_move = action
                formed_mill = env.place_piece(action, ai_player)
            else:
                from_pos, to_pos = action
                print(f"AI moves from position {from_pos} to {to_pos}")
                last_move = to_pos
                formed_mill = env.move_piece(from_pos, to_pos, ai_player)

            
            if formed_mill and env.can_remove(human_player):
                print("\nAI formed a mill and removes one of your pieces!")

                
                removed = False

                
                for pos in range(24):
                    if env.board[pos] == human_player:
                        in_mill = False
                        for mill in env.mills:
                            if pos in mill and all(env.board[m] == human_player for m in mill):
                                in_mill = True
                                break

                        if not in_mill:
                            print(f"AI removes your piece at position {pos}")
                            env.remove_piece(pos, human_player)
                            removed = True
                            break

                
                if not removed:
                    for pos in range(24):
                        if env.board[pos] == human_player:
                            print(f"AI removes your piece at position {pos}")
                            env.remove_piece(pos, human_player)
                            break

        
        winner = env.check_winner()
        if winner > 0:
            print_board(env.board, last_move)
            if winner == human_player:
                print("\nCongratulations! You win!")
            else:
                print("\nAI wins! Better luck next time.")
            done = True
            break

        
        current_player = 3 - current_player

    
    print_board(env.board)
    print("\nGame over!")


def main():
    print("\n===== Nine Men's Morris =====")
    print("A classic board game with AI opponent")

    model_filename = "nine_mens_morris_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(device=device)

    
    if not os.path.exists(model_filename):
        print("\nNo trained model found. Training a new model...")
        print("This might take a few minutes...")
        agent = train_agent(episodes=1000, model_filename=model_filename)
        print("Basic training complete!")
    else:
        print("\nLoading trained model...")
        agent.load(model_filename)

    while True:
        print("\n===== MENU =====")
        print("1. Play against AI")
        print("2. Train AI more (improves performance)")
        print("3. Quit")

        choice = input("\nSelect an option: ")

        if choice == "1":
            print("\nSelect AI difficulty:")
            print("1. Easy")
            print("2. Moderate")
            print("3. Difficult")

            difficulty_choice = input("\nSelect difficulty: ")
            difficulty = "easy"
            if difficulty_choice == "2":
                difficulty = "moderate"
            elif difficulty_choice == "3":
                difficulty = "difficult"

            play_game_vs_ai(agent, difficulty)

        elif choice == "2":
            print("\nTraining AI model...")
            additional_episodes = int(input("\nNumber of additional: "))

            print(f"Training for {additional_episodes} episodes. This might take a few minutes...")
            train_agent(episodes=additional_episodes, model_filename=model_filename, resume=True)
            agent.load(model_filename)  
            print("Training complete!")

        elif choice == "3":
            print("\nThanks for playing!")
            break

        else:
            print("\nInvalid option. Please try again.")

if __name__ == "__main__":
    main()