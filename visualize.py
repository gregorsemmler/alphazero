import numpy as np
import cv2 as cv

from game import Player


def show_im(im):
    cv.imshow("im", im)
    cv.waitKey()
    cv.destroyAllWindows()


def visualization_experiments():
    state_h, state_w = 6, 7
    state = np.random.randint(-1, 2, size=(state_h, state_w))
    probs = np.random.dirichlet([0.5] * state_h)

    entry_size = 100
    entry_radius = entry_size // 2
    margin = 20
    total_w = state_w * (entry_size + margin) + margin
    total_h = state_h * (entry_size + margin) + margin

    no_player_color = (255, 255, 255)
    bg_color = (84, 50, 20)
    # player1_color = (49, 91, 94)
    # player2_color = (33, 30, 83)
    player1_color = (0, 255, 255)
    player2_color = (0, 0, 255)
    viz_img = np.zeros((total_h, total_w, 3), dtype=np.uint8)

    prob_col = (0, 0, 255)  # TODO
    prob_img = np.zeros_like(viz_img)
    prob_img[:, :] = (255, 255, 255)
    prob_w = total_w // state_w
    prob_hs = [int(pr * total_h) for pr in probs]
    for prob_idx, prob_h in enumerate(prob_hs):
        start_pt = (prob_idx * prob_w, total_h - prob_h)
        end_pt = ((prob_idx + 1) * prob_w, total_h-1)
        cv.rectangle(prob_img, start_pt, end_pt, prob_col, thickness=-1)

    show_im(prob_img)

    viz_img[:, :] = bg_color
    for c_idx in range(state_w):
        for r_idx in range(state_h):
            if state[r_idx, c_idx] == Player.NO_PLAYER.value:
                col = no_player_color
            elif state[r_idx, c_idx] == Player.FIRST_PLAYER.value:
                col = player1_color
            else:
                col = player2_color

            cv.circle(viz_img,
                      (margin + c_idx * (entry_size + margin) + entry_radius,
                       margin + r_idx * (entry_size + margin) + entry_radius),
                      entry_radius, col, -1)

            # show_im(viz_img)

            print("")

    show_im(viz_img)
    print("")
    pass


if __name__ == "__main__":
    visualization_experiments()
