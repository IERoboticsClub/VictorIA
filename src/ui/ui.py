import streamlit as st
import numpy as np
import random

from utils import draw, machine_moves, player_moves, is_col_full, winning_move

st.title("Connect 4")
num_columns = 7
num_rows = 6
#Choose level
level = st.radio("Choose level", options=["Easy", "Medium", "Hard"])
st.write(f"You selected {level} level")

if 'board' not in st.session_state:
    st.session_state.board = np.zeros((num_rows, num_columns), dtype=int)

st.session_state.selected_option = st.selectbox("Who starts?", ["Machine", "User"])

if "is_machine_turn" not in st.session_state:
    if st.session_state.selected_option == "Machine":
        machine_moves()
    else:
        st.session_state.is_machine_turn = False

if st.button("Reset"):
    st.session_state.board = np.zeros((num_rows, num_columns), dtype=int)
    if st.session_state.selected_option == "Machine":
        machine_moves()
    else:
        st.session_state.is_machine_turn = False

col = st.number_input("Choose a column (0-6):", 0, 6)



response_container = st.container()

with response_container:

    if st.button("Drop Chip"):
        if is_col_full(st.session_state.board, col):
            st.error("Column is full!")
        else:
            player_moves(col)
            #Check if player won
            if winning_move(st.session_state.board, "player", num_columns, num_rows):
                st.success("You won!")
                draw(num_columns, num_rows)
                st.stop()
            
            machine_moves()
            #Check if machine won
            if winning_move(st.session_state.board, "machine", num_columns, num_rows):
                st.error("You lost!")
                draw(num_columns, num_rows)
                st.stop()

    draw(num_columns, num_rows)
    
