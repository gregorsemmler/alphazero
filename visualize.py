import numpy as np
import cv2 as cv

from game import Player


def show_ims(*ims):
    for im_idx, im in enumerate(ims):
        cv.imshow(f"{im_idx}", im)
    cv.waitKey()
    cv.destroyAllWindows()


def visualization_experiments():
    state_h, state_w = 6, 7
    state = np.random.randint(-1, 2, size=(state_h, state_w))
    probs = np.random.dirichlet([0.5] * state_h)
    viz_im, viz_with_prob = draw_connect_n_state(state, probs)
    show_ims(viz_im, viz_with_prob)


def visualize_connect_n_game(state: np.ndarray, probs):
    viz_im, viz_with_prob = draw_connect_n_state(state, probs)
    show_ims(viz_with_prob)


def draw_connect_n_state(state: np.ndarray, probs):
    state_h, state_w = state.shape

    entry_size = 100
    entry_radius = entry_size // 2
    margin = 20
    total_w = state_w * (entry_size + margin) + margin
    total_h = state_h * (entry_size + margin) + margin

    no_player_color = (255, 255, 255)
    bg_color = (84, 50, 20)
    player1_color = (0, 255, 255)
    player2_color = (0, 0, 255)
    viz_img = np.zeros((total_h, total_w, 3), dtype=np.uint8)

    # prob_col = (0, 0, 255)
    prob_col = (104, 172, 84)
    prob_img = np.zeros_like(viz_img)
    prob_img[:, :] = (255, 255, 255)
    prob_hs = [int(pr * total_h) for pr in probs]
    prob_viz_factor = 0.5

    viz_img[:, :] = bg_color
    prob_ws = []
    previous_border = 0

    for c_idx in range(state_w):
        center_x = margin + c_idx * (entry_size + margin) + entry_radius

        for r_idx in range(state_h):
            if state[r_idx, c_idx] == Player.NO_PLAYER.value:
                col = no_player_color
            elif state[r_idx, c_idx] == Player.FIRST_PLAYER.value:
                col = player1_color
            else:
                col = player2_color

            center_y = margin + r_idx * (entry_size + margin) + entry_radius

            cv.circle(viz_img, (center_x, center_y), entry_radius, col, -1)

        next_border = center_x + entry_radius + margin // 2

        if c_idx < state_w - 1:
            cur_w = next_border - previous_border
        else:
            cur_w = total_w - previous_border
        previous_border = next_border
        prob_ws.append(cur_w)

    last_x = 0
    for prob_idx, (prob_w, prob_h) in enumerate(zip(prob_ws, prob_hs)):
        next_x = last_x + prob_w
        start_pt = (last_x, total_h - prob_h)
        end_pt = (next_x, total_h-1)
        last_x = next_x

        cv.rectangle(prob_img, start_pt, end_pt, prob_col, thickness=-1)

    alpha_prob = np.zeros((total_h, total_w), dtype=np.uint8)
    alpha_prob[:, :] = 255
    alpha_prob[(prob_img == (255, 255, 255)).all(axis=2)] = 0

    alpha2 = (alpha_prob / 255.0) * prob_viz_factor
    alpha1 = 1.0 - alpha2

    viz_img_with_probs = viz_img.copy()
    for c in range(3):
        viz_img_with_probs[:, :, c] = alpha1 * viz_img[:, :, c] + alpha2 * prob_img[:, :, c]

    return viz_img, viz_img_with_probs


if __name__ == "__main__":
    visualization_experiments()
