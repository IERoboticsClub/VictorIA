import streamlit as st
import numpy as np

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
                    if st.session_state.board[acc_row][acc_col] == 1:#player
                        st.markdown(f'<div style="width: 50px; height: 50px; background-color: green; border-radius: 50%; margin: 10px;"></div>',  unsafe_allow_html=True)
                    elif st.session_state.board[acc_row][acc_col] == 2: #machine
                        st.markdown(f'<div style="width: 50px; height: 50px; background-color: red; border-radius: 50%; margin: 10px;"></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div style="width: 50px; height: 50px; background-color: grey; border-radius: 50%; margin: 10px;"></div>', unsafe_allow_html=True)



def machine_moves():
    # random move
    col = np.random.randint(0, 7)
    while is_col_full(st.session_state.board, col):
        col = np.random.randint(0, 7)
        #minimax algorithm

    for row in range(5, -1, -1):
        if st.session_state.board[row][col] == 1 or st.session_state.board[row][col] == 2:
            continue
        else:
            st.session_state.board[row][col] = 2
            st.session_state.is_machine_turn = False
            break


def player_moves(col):
    for row in range(5, -1, -1):
            if st.session_state.board[row][col] == 1 or st.session_state.board[row][col] == 2:
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

def winning_move(board, turn, num_columns, num_rows):
    piece = 1 if turn == "player" else 2

    # Check negatively sloped diaganols
    for c in range(num_columns-3):
        for r in range(3, num_rows):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True
     # Check positively sloped diaganols
    for c in range(num_columns-3):
        for r in range(num_rows-3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True
            
    # Check vertical locations for win
    for c in range(num_columns):
        for r in range(num_rows-3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True
	# Check horizontal locations for win
    for c in range(num_columns-3):
	    for r in range(num_rows):
		    if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
			    return True
              
    
#minimax algorithm