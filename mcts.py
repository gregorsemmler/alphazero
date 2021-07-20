from math import sqrt

import numpy as np
import torch

from game import Player


class MCTS(object):

    def __init__(self, model, game, num_actions, search_batch_size=8, c_puct=1.0, epsilon=0.25, alpha=0.03,
                 device_token="cpu"):
        self.model = model
        self.game = game
        self.num_actions = num_actions
        self.search_batch_size = search_batch_size
        self.alpha = alpha
        self.epsilon = epsilon
        self.c_puct = c_puct
        self.n = {}
        self.w = {}
        self.q = {}
        self.p = {}
        self.device = torch.device(device_token)

    def play_match(self, num_mcts_searches, tau_func, other_mcts=None, first_player=None, replay_buffer=None):
        if other_mcts is None:
            other_mcts = self
        if type(self.game) is not type(other_mcts.game):
            raise RuntimeError("Other MCTS object has game of different type")

        def switch_player(p):
            return Player.FIRST_PLAYER if p == Player.SECOND_PLAYER else Player.SECOND_PLAYER

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

        while True:
            tau = tau_func(step_idx)
            m: MCTS = mcts_d[player]
            m.traversals_and_backups(num_mcts_searches, state, player)
            probs = m.policy_value(state, tau)
            game_history.append((state, player, probs))
            action = np.random.choice(self.num_actions, p=probs)
            if action in m.game.invalid_actions():
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
            for s, p, pr in game_history:
                replay_buffer.append((s, p, pr, final_result if p == player else (-1) * final_result))

        return first_player_result, step_idx

    def is_leaf_state(self, state):
        return state in self.p

    # TODO refactor?
    def traversals_and_backups(self, number_of_searches, state, player):
        for _ in range(number_of_searches):
            self.traverse_and_backup(self.search_batch_size, state, player)

    def traverse_and_backup(self, count, state, player):
        backup_queue = []  # TODO rename?
        planned = set()  # TODO specific state class?
        expand_queue = []

        for _ in range(count):
            value, leaf_state, leaf_player, states, actions = self.traverse(state, player)

            if value is not None:
                backup_queue.append((value, states, actions))
            else:
                if leaf_state not in planned:
                    planned.add(leaf_state)
                    expand_queue.append((leaf_state, states, actions))

        if len(planned) > 0:
            # expand nodes:
            model_in = self.model.state_to_input(planned, self.device)
            prior_probs_out, values_out = self.model.predict(model_in)

            values_np = values_out.detach().cpu().numpy()
            prior_probs_np = prior_probs_out.detach().cpu().numpy()

            for (leaf_state, states, actions), val, prob in zip(expand_queue, values_np, prior_probs_np):
                self.n[leaf_state] = np.zeros((self.num_actions,))
                self.w[leaf_state] = np.zeros((self.num_actions,))
                self.q[leaf_state] = np.zeros((self.num_actions,))
                self.p[leaf_state] = np.zeros((self.num_actions,))
                backup_queue.append((val, states, actions))

        # perform backup
        for val, states, actions in backup_queue:
            # TODO refactor
            cur_value = -val
            for state_int, action in zip(states[::-1], actions[::-1]):
                self.n[state_int][action] += 1
                self.w[state_int][action] += cur_value
                self.q[state_int][action] = self.w[state_int][action] / self.n[state_int][action]
                cur_value = -cur_value

    def traverse(self, state, player):
        states = []
        actions = []
        cur_state = state
        value = None
        cur_player = player
        root_node = True

        while not self.is_leaf_state(state):
            states.append(cur_state)

            visit_counts = self.w[cur_state]
            state_probs = self.p[cur_state]
            state_values = self.q[cur_state]

            if root_node:
                root_node = False
                state_probs = (1 - self.epsilon) * state_probs + self.epsilon * np.random.dirichlet(
                    [self.epsilon] * self.num_actions)

            state_score = state_values + self.c_puct * state_probs * sqrt(sum(visit_counts)) / (1 + visit_counts)

            invalid_actions = self.game.invalid_actions(cur_state)
            state_score[list(invalid_actions)] = -np.inf

            cur_action = int(state_score.argmax())
            cur_state, won = self.game.move(cur_state, cur_action, cur_player)  # TODO refactor?
            if won:
                value = -1.0
            elif len(self.game.valid_actions(cur_state)) == 0:
                # Draw
                value = 0.0

            cur_player = 1 if cur_player == 0 else 1  # TODO refactor?

        # TODO return is_leaf instead of returning Value == None?
        return value, cur_state, cur_player, states, actions

    def policy_value(self, state, tau=1.0):
        counts = self.n[state]
        if tau == 0.0:
            result = np.zeros((self.num_actions,))
            result[counts.argmax()] = 1.0
            return result

        sum_counts = counts.sum()
        return np.array([count ** (1.0 / tau) / sum_counts for count in counts])
