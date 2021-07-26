from math import sqrt

import numpy as np
import torch
import torch.nn.functional as F

from game import Player, switch_player, Game


class TreeNode(object):

    def __init__(self, state, player, value=None, states_visited=(), actions_taken=()):
        self.state = state
        self.player = player
        self.value = value
        self.states_visited = states_visited
        self.actions_taken = actions_taken

    def __str__(self):
        return f"{self.state} - {self.player} - {self.states_visited} - {self.actions_taken}"


class GameHistoryEntry(object):

    def __init__(self, state, player, probs, value=None):
        self.state = state
        self.player = player
        self.probs = probs
        self.value = value

    def __str__(self):
        return f"{self.state} - {self.player} - {self.probs} - {self.value}"


class MonteCarloTreeSearch(object):

    def __init__(self, model, game: Game, search_batch_size=8, c_puct=1.0, epsilon=0.25, alpha=0.03,
                 device=torch.device("cpu")):
        self.model = model
        self.game = game
        self.num_actions = game.num_actions
        self.search_batch_size = search_batch_size
        self.alpha = alpha
        self.epsilon = epsilon
        self.c_puct = c_puct
        self.n = {}
        self.w = {}
        self.q = {}
        self.p = {}
        self.device = device

    def reset(self):
        self.n = {}
        self.w = {}
        self.q = {}
        self.p = {}

    def play_match(self, num_mcts_searches, tau_func, other_mcts=None, first_player=None, replay_buffer=None):
        if other_mcts is None:
            other_mcts = self
        if type(self.game) is not type(other_mcts.game):
            raise RuntimeError("Other MCTS object has game of different type")

        state = self.game.initial_state()
        first_player = first_player if first_player is not None else np.random.choice(
            [Player.FIRST_PLAYER, Player.SECOND_PLAYER])
        other_player = switch_player(first_player)
        player = first_player
        step_idx = 0
        game_history = []

        mcts_d = {first_player: self, other_player: other_mcts}
        first_player_result = None
        final_result = None

        self.model.eval()
        other_mcts.model.eval()

        while True:
            tau = tau_func(step_idx)
            m: MonteCarloTreeSearch = mcts_d[player]
            m.traverse_and_backup(num_mcts_searches, state, player)
            probs = m.policy_value(state, tau)
            game_history.append(GameHistoryEntry(state, player, probs))
            action = np.random.choice(self.num_actions, p=probs)
            if action in m.game.invalid_actions(state):
                raise RuntimeError("Impossible Action selected")
            state, won = m.game.move(state, action, player)

            if won:
                final_result = 1
                first_player_result = final_result if player == first_player else (-1) * final_result
            elif len(m.game.valid_actions(state)) == 0:
                # draw
                final_result = 0
                first_player_result = final_result

            if final_result is not None:
                break

            player = switch_player(player)
            step_idx += 1

        if replay_buffer is not None:
            for entry in game_history:
                entry.value = final_result if entry.player == player else (-1) * final_result
                replay_buffer.append(entry)

        return first_player_result, step_idx+1

    def is_leaf_state(self, state):
        return self.game.encode_state(state) not in self.p

    def traverse_and_backup(self, number_of_simulations, state, player):
        backup_queue = []
        planned = set()
        expand_queue = []

        for _ in range(number_of_simulations):
            for _ in range(self.search_batch_size):
                leaf_node = self.traverse(state, player)

                if leaf_node.value is not None:
                    backup_queue.append(leaf_node)
                else:
                    leaf_state_h = self.game.encode_state(leaf_node.state)
                    if leaf_state_h not in planned:
                        planned.add(leaf_state_h)
                        expand_queue.append(leaf_node)

            if len(planned) > 0:
                # expand nodes:
                model_in = torch.stack([self.game.state_to_tensor(el.state, device=self.device) for el in expand_queue])

                with torch.no_grad():
                    log_probs_out, values_out = self.model(model_in)
                    prior_probs_out = F.softmax(log_probs_out, dim=1)

                values_np = values_out.detach().cpu().numpy()
                prior_probs_np = prior_probs_out.detach().cpu().numpy()

                for idx, (node, pred_val, pred_prob) in enumerate(
                        zip(expand_queue, values_np, prior_probs_np)):
                    state_h = self.game.encode_state(node.state)
                    self.n[state_h] = np.zeros((self.num_actions,))
                    self.w[state_h] = np.zeros((self.num_actions,))
                    self.q[state_h] = np.zeros((self.num_actions,))
                    self.p[state_h] = pred_prob
                    node.value = pred_val
                    backup_queue.append(node)

            # perform backup
            for node in backup_queue:
                cur_value = -node.value
                for state, action in zip(reversed(node.states_visited), reversed(node.actions_taken)):
                    state_h = self.game.encode_state(state)
                    self.n[state_h][action] += 1
                    self.w[state_h][action] += cur_value
                    self.q[state_h][action] = self.w[state_h][action] / self.n[state_h][action]
                    cur_value = -cur_value

    def traverse(self, state, player):
        visited_states = []
        taken_actions = []
        cur_state = state
        value = None
        cur_player = player
        root_node = True

        while not self.is_leaf_state(cur_state):
            visited_states.append(cur_state)
            cur_state_h = self.game.encode_state(cur_state)

            visit_counts = self.n[cur_state_h]
            state_probs = self.p[cur_state_h]
            state_values = self.q[cur_state_h]

            if root_node:
                root_node = False
                state_probs = (1 - self.epsilon) * state_probs + self.epsilon * np.random.dirichlet(
                    [self.alpha] * self.num_actions)

            state_score = state_values + self.c_puct * state_probs * sqrt(sum(visit_counts)) / (1 + visit_counts)

            invalid_actions = self.game.invalid_actions(cur_state)
            state_score[list(invalid_actions)] = -np.inf

            cur_action = int(state_score.argmax())
            taken_actions.append(cur_action)

            cur_state, won = self.game.move(cur_state, cur_action, cur_player)
            if won:
                value = -1.0
            elif len(self.game.valid_actions(cur_state)) == 0:
                # Draw
                value = 0.0

            cur_player = switch_player(cur_player)

        return TreeNode(cur_state, cur_player, value, visited_states, taken_actions)

    def policy_value(self, state, tau=1.0):
        state_h = self.game.encode_state(state)
        counts = self.n[state_h]
        if tau == 0.0:
            result = np.zeros((self.num_actions,))
            result[counts.argmax()] = 1.0  # TODO tie breaking?
            return result

        sum_counts = counts.sum()
        return np.array([count ** (1.0 / tau) / sum_counts for count in counts])
