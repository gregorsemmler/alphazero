from math import sqrt

import numpy as np
import torch
import torch.nn.functional as F

from game import Player, switch_player, Game


class MCTS(object):

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
            m: MCTS = mcts_d[player]
            m.traversals_and_backups(num_mcts_searches, state, player)
            probs = m.policy_value(state, tau)
            game_history.append((state, player, probs))
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
            for st, pl, prb in game_history:
                replay_buffer.append((st, prb, final_result if pl == player else (-1) * final_result))

        return first_player_result, step_idx

    def is_leaf_state(self, state):
        return self.game.encode_state(state) not in self.p

    # TODO refactor?
    def traversals_and_backups(self, number_of_searches, state, player):
        for _ in range(number_of_searches):
            self.traverse_and_backup(self.search_batch_size, state, player)

    def traverse_and_backup(self, count, state, player):
        backup_queue = []  # TODO rename?
        planned = {}
        expand_queue = []

        for _ in range(count):
            value, leaf_state, leaf_player, states, actions = self.traverse(state, player)

            if value is not None:
                backup_queue.append((value, states, actions))
            else:
                leaf_state_h = self.game.encode_state(leaf_state)
                if leaf_state_h not in planned:
                    planned[leaf_state_h] = leaf_state
                    expand_queue.append((leaf_state, states, actions))

        if len(planned) > 0:
            # expand nodes:
            model_in = torch.stack([self.game.state_to_tensor(el, device=self.device) for k, el in planned.items()])

            with torch.no_grad():
                log_probs_out, values_out = self.model(model_in)
                prior_probs_out = F.softmax(log_probs_out, dim=1)

            values_np = values_out.detach().cpu().numpy()
            prior_probs_np = prior_probs_out.detach().cpu().numpy()

            for idx, ((leaf_state, states, actions), val, prob) in enumerate(
                    zip(expand_queue, values_np, prior_probs_np)):
                # TODO refactor
                state_h = self.game.encode_state(leaf_state)
                self.n[state_h] = np.zeros((self.num_actions,))
                self.w[state_h] = np.zeros((self.num_actions,))
                self.q[state_h] = np.zeros((self.num_actions,))
                self.p[state_h] = prior_probs_np[idx]
                backup_queue.append((val, states, actions))

        # perform backup
        for val, states, actions in backup_queue:
            # TODO refactor
            cur_value = -val
            for state, action in zip(reversed(states), reversed(actions)):
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

            cur_state, won = self.game.move(cur_state, cur_action, cur_player)  # TODO refactor?
            if won:
                value = -1.0
            elif len(self.game.valid_actions(cur_state)) == 0:
                # Draw
                value = 0.0

            cur_player = switch_player(cur_player)

        # TODO return is_leaf instead of returning Value == None?
        return value, cur_state, cur_player, visited_states, taken_actions

    def policy_value(self, state, tau=1.0):
        state_h = self.game.encode_state(state)
        counts = self.n[state_h]
        if tau == 0.0:
            result = np.zeros((self.num_actions,))
            result[counts.argmax()] = 1.0  # TODO tie breaking?
            return result

        sum_counts = counts.sum()
        return np.array([count ** (1.0 / tau) / sum_counts for count in counts])
