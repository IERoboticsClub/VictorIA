class ConnectFour:
    """
    ConnectFour class represents the Connect Four game.

    Attributes:
        - rows (int): The number of rows in the game board.
        - columns (int): The number of columns in the game board.
        - board (list): The game board represented as a 2D list.
        - turn (str): The current player's turn ("X" or "O").
        - depth (int): The depth for the minimax algorithm.

    Methods:
        - insert_disc(column, player): Inserts a disc into the specified column for the given player.
        - check_winner(player): Checks if the given player has won the game.
        - greedy_move(): Determines the best move using a greedy strategy.
        - minimax_move(): Determines the best move using the minimax algorithm.
        - minimax(depth, maximizing_player): Performs the minimax algorithm to evaluate the game board.
        - get_next_open_row(col): Returns the next open row in the specified column.
        - is_col_full(col): Checks if the specified column is full.
        - is_board_full(): Checks if the game board is full.
        - evaluate_board(board): Evaluates the game board and returns a score.
        - score_position(position): Scores a position based on the number of "O" and "X" in it.
        - simulate_move(col, player): Simulates a move by inserting a disc into the specified column for the given player.
    """

    def __init__(self):
        self.rows = 6
        self.columns = 7
        self.board = [[" " for _ in range(self.columns)] for _ in range(self.rows)]
        self.turn = "X"  # X will start
        self.depth = 3  # Depth for the minimax algorithm

    def insert_disc(self, column, player):
        """
        Inserts a disc into the specified column for the given player.

        Args:
            - column (int): The column to insert the disc into.
            - player (str): The player ("X" or "O") who is inserting the disc.

        Returns:
            - bool: True if the disc was successfully inserted, False otherwise.

        O(n) time complexity
        """
        if self.board[0][column] != " ":
            return False  # Column is full

        for row in range(self.rows - 1, -1, -1):
            if self.board[row][column] == " ":
                self.board[row][column] = player
                return True
        return False

    def check_winner(self, player):
        """
        Checks for a win condition in four different ways, corresponding to horizontal, vertical, and two diagonal directions.

        Args:
            - player (str): The player ("X" or "O") to check for a win.

        Returns:
            - bool: True if the player has won, False otherwise.

        O(n^2) time complexity
        """
        # Check horizontal, vertical, and diagonals for a win
        for row in range(self.rows):
            for col in range(self.columns - 3):
                if all(self.board[row][c] == player for c in range(col, col + 4)):
                    return True

        for col in range(self.columns):
            for row in range(self.rows - 3):
                if all(self.board[r][col] == player for r in range(row, row + 4)):
                    return True

        for row in range(self.rows - 3):
            for col in range(self.columns - 3):
                if all(self.board[row + i][col + i] == player for i in range(4)):
                    return True

        for row in range(3, self.rows):
            for col in range(self.columns - 3):
                if all(self.board[row - i][col + i] == player for i in range(4)):
                    return True

        return False

    def greedy_move(self):
        """
        Determines the best move using a greedy strategy.

        Returns:
            - int: The column index of the best move.

        O(n^2) time complexity
        """
        # CHeck if the player can win in the next move
        for col in range(self.columns):
            if self.simulate_move(col, "O"):
                return col

        # Block the other player from winning
        for col in range(self.columns):
            if self.simulate_move(col, "X"):
                return col

        # Fallback strategy: choose the center column or the closest available column using offset
        for offset in range(self.columns // 2 + 1):
            for col in [self.columns // 2 + offset, self.columns // 2 - offset]:
                if 0 <= col < self.columns and self.board[0][col] == " ":
                    return col

    def minimax_move(self):
        """
        Determines the best move using the minimax algorithm.

        Returns:
            - int: The column index of the best move.

        O(n) time complexity
        """
        best_score = float("-inf")
        best_move = None

        for col in range(self.columns):
            if not self.is_col_full(col):
                row = self.get_next_open_row(col)
                self.insert_disc(col, "O")
                score = self.minimax(self.depth, False)
                self.board[row][col] = " "  # Undo the move

                if score > best_score:
                    best_score = score
                    best_move = col

        return best_move

    def minimax(self, depth, maximizing_player):
        """
        Performs the minimax algorithm to evaluate the game board.

        Args:
            - depth (int): The current depth of the algorithm.
            - maximizing_player (bool): True if the current player is maximizing, False otherwise.

        Returns:
            - int: The score of the evaluated game board.

        O(b^m) time complexity, where b is the branching factor and m is the maximum depth of the search tree; in this case, b = 7 and m = 3
        """
        if (
            depth == 0
            or self.check_winner("O")
            or self.check_winner("X")
            or self.is_board_full()
        ):
            return self.evaluate_board(self.board)

        if maximizing_player:
            max_eval = float("-inf")
            for col in range(self.columns):
                if not self.is_col_full(col):
                    row = self.get_next_open_row(col)
                    self.insert_disc(col, "O")
                    eval = self.minimax(depth - 1, False)
                    self.board[row][col] = " "  # Undo the move
                    max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float("inf")
            for col in range(self.columns):
                if not self.is_col_full(col):
                    row = self.get_next_open_row(col)
                    self.insert_disc(col, "X")
                    eval = self.minimax(depth - 1, True)
                    self.board[row][col] = " "  # Undo the move
                    min_eval = min(min_eval, eval)
            return min_eval

    def get_next_open_row(self, col):
        """
        Returns the next open row in the specified column.

        Args:
            - col (int): The column index.

        Returns:
            - int: The row index of the next open row.

        O(n) time complexity
        """
        # Start from the bottom of the column and move upward
        for row in range(self.rows - 1, -1, -1):
            if self.board[row][col] == " ":
                return row

    def is_col_full(self, col):
        """
        Checks if the specified column is full.

        Args:
            - col (int): The column index.

        Returns:
            - bool: True if the column is full, False otherwise.

        O(1) time complexity
        """
        return self.board[0][col] != " "

    def is_board_full(self):
        """
        Checks if the game board is full.

        Returns:
            - bool: True if the game board is full, False otherwise.

        O(n^2) time complexity
        """
        return all(cell != " " for row in self.board for cell in row)

    def evaluate_board(self, board):
        """
        Evaluates the game board and returns a score.

        Args:
            - board (list): The game board represented as a 2D list.

        Returns:
            - int: The score of the game board.

        O(n^2) time complexity (same as check_winner)
        """
        # Check for a win
        if self.check_winner("O"):
            return 10  # "O" wins, maximizing player
        elif self.check_winner("X"):
            return -10  # "X" wins, minimizing player

        # Count the number of open two-in-a-row, three-in-a-row, and four-in-a-row for each player
        score = 0

        # Check rows
        for row in range(self.rows):
            for col in range(self.columns - 3):
                score += self.score_position(board[row][col : col + 4])

        # Check columns
        for col in range(self.columns):
            for row in range(self.rows - 3):
                score += self.score_position([board[row + i][col] for i in range(4)])

        # Check diagonals
        for row in range(self.rows - 3):
            for col in range(self.columns - 3):
                score += self.score_position(
                    [board[row + i][col + i] for i in range(4)]
                )

        for row in range(3, self.rows):
            for col in range(self.columns - 3):
                score += self.score_position(
                    [board[row - i][col + i] for i in range(4)]
                )

        return score

    def score_position(self, position):
        """
        Scores a position based on the number of "O" and "X" in it.

        Args:
            - position (list): The position to score.

        Returns:
            - int: The score of the position.

        O(1) time complexity
        """
        # Assign scores based on the number of "O" and "X" in a position
        o_count = position.count("O")
        x_count = position.count("X")

        if o_count == 4:
            return 100
        elif o_count == 3 and x_count == 0:
            return 5
        elif o_count == 2 and x_count == 0:
            return 2
        elif x_count == 4:
            return -100
        elif x_count == 3 and o_count == 0:
            return -5
        elif x_count == 2 and o_count == 0:
            return -2

        return 0

    def simulate_move(self, col, player):
        """
        Simulates a move by inserting a disc into the specified column for the given player.

        Args:
            - col (int): The column index.
            - player (str): The player ("X" or "O") who is simulating the move.

        Returns:
            - bool: True if the move results in a win, False otherwise.

        O(n) time complexity
        """
        for row in range(self.rows - 1, -1, -1):
            if self.board[row][col] == " ":
                self.board[row][col] = player
                win = self.check_winner(player)
                self.board[row][col] = " "  # Undo the move
                if win:
                    return True
                break
        return False
