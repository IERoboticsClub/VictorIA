import pandas as pd
import streamlit as st
from utils import ConnectFour

if "users_df" not in st.session_state:
    st.session_state.users_df = pd.DataFrame(
        columns=["Username", "Wins", "Losses", "Win/Loss Ratio"]
    )


# Function to draw the board
def draw(num_columns, num_rows):
    """
    Draws the game board with circles representing the player and machine moves.

    Args:
        - num_columns (int): The number of columns in the game board.
        - num_rows (int): The number of rows in the game board.

    Returns:
        - None

    O(n*m) time complexity, where n is the number of columns and m is the number of rows.
    """
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
                    if st.session_state.game.board[acc_row][acc_col] == "X":  # player
                        st.markdown(
                            f'<div style="width: 50px; height: 50px; '
                            f'background-color: green; border-radius: 50%; margin: 10px;"></div>',
                            unsafe_allow_html=True,
                        )
                    elif (
                        st.session_state.game.board[acc_row][acc_col] == "O"
                    ):  # machine
                        st.markdown(
                            f'<div style="width: 50px; height: 50px; '
                            f'background-color: red; border-radius: 50%; margin: 10px;"></div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div style="width: 50px; height: 50px; '
                            f'background-color: grey; border-radius: 50%; margin: 10px;"></div>',
                            unsafe_allow_html=True,
                        )


def machine_moves(game, player="O", move_type="greedy"):
    """
    Makes a move for the machine player in the game.

    Args:
        - game (Game): The game object representing the current state of the game.
        - player (str): The player symbol for the machine player. Default is "O".
        - move_type (str): The type of move to be made. Valid options are "greedy" and "minimax". Default is "greedy".

    Raises:
        - ValueError: If an invalid move type is provided.

    Returns:
        - None

    O(b^m) time complexity for minimax (worst case), where b is the branching factor and m is the maximum depth of the tree.
    """
    if move_type == "greedy":
        col = game.greedy_move()
    elif move_type == "minimax":
        col = game.minimax_move()
    else:
        raise ValueError("Invalid move type. Choose 'greedy' or 'minimax'.")

    game.insert_disc(col, player)


# Function to check if a column is full
def is_col_full(board, col):
    """
    Checks if the specified column is full.

    Args:
        - col (int): The column index.

    Returns:
        - bool: True if the column is full, False otherwise.

    O(1) time complexity
    """
    return board[0][col] != " "


# Function to process player moves
def player_moves(game, col, player="X"):
    """
    Insert a disc for the specified player in the given column of the game board.

    Args:
        - game (Game): The game object representing the current state of the game.
        - col (int): The column number where the disc should be inserted.
        - player (str, optional): The player symbol. Defaults to "X".

    Returns:
        - None

    O(1) time complexity
    """
    if not is_col_full(game.board, col):
        game.insert_disc(col, player)


def main_loop():
    """
    Main loop for the Connect 4 game.

    This function handles the gameplay logic, including initializing the game board,
    getting player usernames, checking if the user already exists in the DataFrame,
    choosing who starts, selecting AI move type, resetting the game, getting player input,
    checking for a winner, updating the user's win/loss ratio, and drawing the game board.

    Returns:
        - str: The username of the winning player, or None if there is no winner.
    """
    st.title("Connect 4")
    columns = 7
    rows = 6

    # Initialize the game board and game_over flag
    if "game" not in st.session_state:
        st.session_state.game = ConnectFour()
        st.session_state.game_over = False

    # Player usernames
    username = st.text_input("Enter username:")
    if username == "":
        st.warning("Please enter a username.")
        st.stop()

    # Check if the user already exists in the DataFrame
    if username not in st.session_state.users_df["Username"].values:
        # Add a new row for the user
        new_user_row = pd.DataFrame(
            [[username, 0, 0, 0]],
            columns=["Username", "Wins", "Losses", "Win/Loss Ratio"],
        )
        st.session_state.users_df = pd.concat(
            [st.session_state.users_df, new_user_row], ignore_index=True
        )
        st.session_state.users_df = st.session_state.users_df.sort_values(
            by="Win/Loss Ratio", ascending=False
        )

    # Choose who starts
    st.session_state.selected_option = st.selectbox("Who starts?", ["Machine", "User"])

    # Choose AI move type
    move_type = st.radio("Select AI Move Type", ["Greedy", "Minimax"])

    # # Reset the game
    if st.button("Reset"):
        st.session_state.game = ConnectFour()
        st.session_state.game_over = False
        if st.session_state.selected_option == "Machine":
            machine_moves(st.session_state.game, move_type=move_type.lower())

    # Player input
    col = st.number_input("Choose a column (0-6):", 0, columns - 1, format="%d")

    response_container = st.container()
    with response_container:
        if st.button("Drop Chip"):
            if st.session_state.game_over:
                st.warning("Game over. Please reset the game to play again.")
            elif is_col_full(st.session_state.game.board, col):
                st.error("Column is full!")
            else:
                player_moves(st.session_state.game, col)
                user_index = st.session_state.users_df[
                    st.session_state.users_df["Username"] == username
                ].index[0]
                # Check if player won
                if st.session_state.game.check_winner("X"):
                    draw(columns, rows)
                    st.success(f"{username} won!")
                    st.session_state.users_df.at[user_index, "Wins"] += 1
                    st.session_state.game_over = True
                    return username

                if not st.session_state.game_over:
                    machine_moves(st.session_state.game, move_type=move_type.lower())

                    # Check if machine won
                    if st.session_state.game.check_winner("O"):
                        st.error("You lost!")
                        st.session_state.users_df.at[user_index, "Losses"] += 1
                        draw(columns, rows)
                        st.session_state.game_over = True
                        return None

                st.session_state.users_df.at[user_index, "Win/Loss Ratio"] = (
                    st.session_state.users_df.at[user_index, "Wins"]
                    / st.session_state.users_df.at[user_index, "Losses"]
                    if st.session_state.users_df.at[user_index, "Losses"] > 0
                    else float("inf")
                )

        draw(columns, rows)


##NB This is the main function that runs the front end separately from cv
main_loop()
