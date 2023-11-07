import numpy as np
import streamlit as st


def draw(num_columns, num_rows):
    st.title("Board")
    columns = st.columns(num_columns)
    acc_col = -1
    acc_row = -1
    while acc_col < 6:
        for col in columns:
            if acc_row > 4:
                acc_row = -1

            acc_col += 1
            with col:
                for circle in range(num_rows):
                    acc_row += 1
                    if st.session_state.board[acc_row][acc_col] == 1:  # player
                        st.markdown(
                            f'<div style="width: 50px; height: 50px; background-color: green; border-radius: 50%; margin: 10px;"></div>',
                            unsafe_allow_html=True,
                        )
                    elif st.session_state.board[acc_row][acc_col] == 2:  # machine
                        st.markdown(
                            f'<div style="width: 50px; height: 50px; background-color: red; border-radius: 50%; margin: 10px;"></div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div style="width: 50px; height: 50px; background-color: grey; border-radius: 50%; margin: 10px;"></div>',
                            unsafe_allow_html=True,
                        )


def machine_moves():
    # random move
    col = minimax(st.session_state.board, 5, -np.Inf, np.Inf, True)[0]

    for row in range(5, -1, -1):
        if (
            st.session_state.board[row][col] == 1
            or st.session_state.board[row][col] == 2
        ):
            continue
        else:
            st.session_state.board[row][col] = 2
            st.session_state.is_machine_turn = False
            break


def player_moves(col):
    for row in range(5, -1, -1):
        if (
            st.session_state.board[row][col] == 1
            or st.session_state.board[row][col] == 2
        ):
            pass
        else:
            st.session_state.board[row][col] = 1
            st.session_state.is_machine_turn = True
            break


def is_col_full(board, col):
    for row in range(5, -1, -1):
        if board[row][col] == 1 or board[row][col] == 2:
            pass
        else:
            return False
    return True


def winning_move(board, turn, num_columns=7, num_rows=6):
    piece = 1 if turn == "player" else 2

    # Check negatively sloped diaganols
    for c in range(num_columns - 3):
        for r in range(3, num_rows):
            if (
                board[r][c] == piece
                and board[r - 1][c + 1] == piece
                and board[r - 2][c + 2] == piece
                and board[r - 3][c + 3] == piece
            ):
                return True
    # Check positively sloped diaganols
    for c in range(num_columns - 3):
        for r in range(num_rows - 3):
            if (
                board[r][c] == piece
                and board[r + 1][c + 1] == piece
                and board[r + 2][c + 2] == piece
                and board[r + 3][c + 3] == piece
            ):
                return True

    # Check vertical locations for win
    for c in range(num_columns):
        for r in range(num_rows - 3):
            if (
                board[r][c] == piece
                and board[r + 1][c] == piece
                and board[r + 2][c] == piece
                and board[r + 3][c] == piece
            ):
                return True
    # Check horizontal locations for win
    for c in range(num_columns - 3):
        for r in range(num_rows):
            if (
                board[r][c] == piece
                and board[r][c + 1] == piece
                and board[r][c + 2] == piece
                and board[r][c + 3] == piece
            ):
                return True


def is_valid_location(board, col):
    return board[5][col] == 0


def get_next_open_row(board, col):
    for r in range(6):
        if board[r][col] == 0:
            return r


def evaluate(board):
    score = 0
    # Score horizontal
    for r in range(6):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(4):
            window = row_array[c : c + 4]
            score += evaluate_window(window)

    # Score vertical
    for c in range(7):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(3):
            window = col_array[r : r + 4]
            score += evaluate_window(window)

    # Score positive sloped diagonal
    for r in range(3):
        for c in range(4):
            window = [board[r + i][c + i] for i in range(4)]
            score += evaluate_window(window)

    # Score negative sloped diagonal
    for r in range(3, 6):
        for c in range(4):
            window = [board[r - i][c + i] for i in range(4)]
            score += evaluate_window(window)

    return score


def score_position(board):
    score = 0
    # Score center column
    center_array = [int(i) for i in list(board[:, 3])]
    center_count = center_array.count(2)
    score += center_count * 3

    # Score Horizontal
    for r in range(6):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(4):
            window = row_array[c : c + 4]
            score += evaluate_window(window)

    # Score Vertical
    for c in range(7):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(3):
            window = col_array[r : r + 4]
            score += evaluate_window(window)

    # Score posiive sloped diagonal
    for r in range(3):
        for c in range(4):
            window = [board[r + i][c + i] for i in range(4)]
            score += evaluate_window(window)

    # Score negative sloped diagonal
    for r in range(3, 6):
        for c in range(4):
            window = [board[r - i][c + i] for i in range(4)]
            score += evaluate_window(window)

    return score


def evaluate_window(window):
    score = 0
    ai_piece = 2
    opp_piece = 1

    if window.count(ai_piece) == 4:
        score += 100
    elif window.count(ai_piece) == 3 and window.count(0) == 1:
        score += 10
    elif window.count(ai_piece) == 2 and window.count(0) == 2:
        score += 5
    elif window.count(ai_piece) == 1 and window.count(0) == 3:
        score += 1

    if window.count(opp_piece) == 3 and window.count(0) == 1:
        score -= 80

    return score


def minimax(board, depth, alpha, beta, maximizingPlayer):
    if depth == 0:
        return -1, evaluate(board)

    if winning_move(board, 2):
        return -1, 100000000000000
    elif winning_move(board, 1):
        return -1, -10000000000000

    valid_locations = [c for c in range(7) if is_valid_location(board, c)]

    if len(valid_locations) == 0 or depth == 0:
        return -1, score_position(board)

    if maximizingPlayer:
        value = -np.Inf
        column = np.random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, 2)
            new_score = minimax(b_copy, depth - 1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value

    else:  # Minimizing player
        value = np.Inf
        column = np.random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, 1)
            new_score = minimax(b_copy, depth - 1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value


def drop_piece(board, row, col, piece):
    board[row][col] = piece
