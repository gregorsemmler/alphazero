import numpy as np


def main():
    # TODO connect N diagonal check test
    n_rows, n_cols = 6, 7
    ar = np.random.randint(-1, 2, size=(n_rows, n_cols))
    row_idx, col_idx = 1, 1
    count_to_win = 4
    move_upper = min(row_idx, col_idx, count_to_win-1)
    move_lower = min(n_rows-row_idx-1, n_cols-col_idx-1, count_to_win-1)

    for offset in range(-1 * move_upper, move_lower-(count_to_win-1)+1):
        print([(row_idx+offset+i, col_idx+offset+i) for i in range(count_to_win)])
        print([ar[row_idx+offset+i, col_idx+offset+i] for i in range(count_to_win)])

    # Diagonal in other direction
    move_upper2 = min(n_cols-col_idx-1, row_idx, count_to_win-1)
    move_lower2 = min(col_idx, n_rows-row_idx-1, count_to_win-1)

    print("=========")

    for offset2 in range(-1 * move_lower2, move_upper2-(count_to_win-1)):
        print([(row_idx+(-1)*offset2+(-1)*i, col_idx+offset2+i) for i in range(count_to_win)])
        print([ar[row_idx+(-1)*offset2+(-1)*i, col_idx+offset2+i] for i in range(count_to_win)])

    print("")

    # diagonal

    print("")


if __name__ == "__main__":
    main()
