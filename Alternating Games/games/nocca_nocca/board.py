import numpy as np
from typing import TypeAlias, Self

BLACK = 0
WHITE = 1
EMPTY = -1
BLACK_START = 1
WHITE_START = 6
BLACK_GOAL = 7
WHITE_GOAL = 0
ROWS = 8
COLS = 5
MAX_STACK = 3
MOVES = ["N", "S", "W", "E", "SW", "NW", "SE", "NE"]

Player: TypeAlias = int
Action: TypeAlias = tuple[int, int, str]
Coords: TypeAlias = tuple[int, int]

class Board:

    def __init__(self):
        self.squares = np.ndarray((ROWS, COLS, MAX_STACK))
        self.squares.fill(EMPTY)
        for y in range(COLS):
            self.squares[BLACK_START][y][0] = BLACK
            self.squares[WHITE_START][y][0] = WHITE

    @staticmethod
    def _opponent(player: Player) -> Player:
        if player == BLACK:
            return WHITE
        return BLACK

    def player_min_distance(self, player: Player) -> float:
        player_squares = np.argwhere(self.squares == player)
        goal = BLACK_GOAL if player == BLACK else WHITE_GOAL
        min_distance = float("inf")
        for x, y, k in player_squares:
            stack = self.squares[x][y]
            index_top = (stack != EMPTY).nonzero()[0].max()

            # bloqueada
            if index_top > k:
                continue

            distance = abs(goal-x)

            if distance < min_distance:
                min_distance = distance

        return min_distance
    
    def blocked_by_opponent(self, player: Player) -> float:
        player_squares = np.argwhere(self.squares == player)
        blocked = 0
        for x, y, k in player_squares:
            stack = self.squares[x][y]
            opponent_pieces = np.argwhere(stack == Board._opponent(player)).tolist()
            player_blocked = len(opponent_pieces) != 0 and any(
                h > k for h in opponent_pieces
            )
            if player_blocked:
                blocked += 1

        return blocked

    def _check_player_blocked(self, player: Player) -> bool:
        player_squares = np.argwhere(self.squares == player)
        for x, y, k in player_squares:
            stack = self.squares[x][y]
            opponent_pieces = np.argwhere(stack == Board._opponent(player)).tolist()
            player_blocked = len(opponent_pieces) != 0 and any(
                h > k for h in opponent_pieces
            )

            if not player_blocked:
                return False
        return True

    @staticmethod
    def _map_action_to_new_pos(action: Action) -> Coords:
        (x, y, move) = action
        match move:
            case "N":
                return x - 1, y
            case "S":
                return x + 1, y
            case "E":
                return x, y + 1
            case "W":
                return x, y - 1
            case "NE":
                return x - 1, y + 1
            case "NW":
                return x - 1, y - 1
            case "SE":
                return x + 1, y + 1
            case "SW":
                return x + 1, y - 1

    def check_game_over(self) -> bool:
        return self.check_for_winner() is not None

    def check_for_winner(self) -> Player:
        # check if a white piece reached the goal or if all black pieces are blocked
        if any(
            x == WHITE for x in self.squares[WHITE_GOAL][:].T[0]
        ) or self._check_player_blocked(BLACK):
            return WHITE
        # check if a black piece reached the goal or if all white pieces are blocked
        elif any(
            x == BLACK for x in self.squares[BLACK_GOAL][:].T[0]
        ) or self._check_player_blocked(WHITE):
            return BLACK
        else:
            return None

    def play_turn(self, player: Player, action: Action) -> None:
        (x, y, _) = action
        # take the highest player piece off the tower
        if self.squares[x][y][2] == player:
            self.squares[x][y][2] = EMPTY
        elif self.squares[x][y][1] == player:
            self.squares[x][y][1] = EMPTY
        else:
            self.squares[x][y][0] = EMPTY

        # put the piece in the correct square
        new_x, new_y = Board._map_action_to_new_pos(action)
        if self.squares[new_x][new_y][0] == EMPTY:
            self.squares[new_x][new_y][0] = player
        elif self.squares[new_x][new_y][1] == EMPTY:
            self.squares[new_x][new_y][1] = player
        else:
            self.squares[new_x][new_y][2] = player

    def legal_moves(self, player: Player) -> list[Action]:
        legal_moves = []
        player_squares = np.argwhere(self.squares == player)
        for square in player_squares:
            for move in MOVES:
                action = (square[0], square[1], move)
                is_legal_move, _ = self.is_legal_move(player, action)
                if is_legal_move:
                    legal_moves.append(action)
        return legal_moves

    def is_legal_move(self, player: Player, action: Action) -> tuple[bool, str]:
        (x, y, move) = action
        # check if there is a piece in position x, y
        stack = self.squares[x][y]
        if all(x != player for x in stack):
            return (False, f"There are no player pieces in position ({x},{y})")
        # check if the piece is blocked
        player_squares = np.argwhere(stack == player).tolist()
        opponent_squares = np.argwhere(stack == Board._opponent(player)).tolist()
        if player_squares != []:
            max_pos = max(player_squares)
            if any(h > max_pos for h in opponent_squares):
                return (
                    False,
                    f"Player pieces in position ({x}, {y}) are blocked by an opponent piece",
                )
        # check if move is legal
        if x in [WHITE_GOAL, BLACK_GOAL]:
            return (False, "Game already over")
        if (x == BLACK_START and player == BLACK and move in ["N", "NW", "NE"]) or (
            x == WHITE_START and player == WHITE and move in ["S", "SW", "SE"]
        ):
            return (False, "Cannot move into your own goal")
        if (y == 0 and move in ["W", "NW", "SW"]) or (
            y == 4 and move in ["E", "NE", "SE"]
        ):
            return (False, "Cannot move out of bounds")
        # check for destination tower height
        new_x, new_y = Board._map_action_to_new_pos(action)
        if self.squares[new_x][new_y][2] != EMPTY:
            return (False, "Cannot move to a full tower")
        # if all checks passed, the move is legal
        return (True, "Legal move")

    def set_board(self, board: Self) -> None:
        self.squares = np.copy(board.squares)
    
    def render(self):
        # rendering a stack of pieces
        def stack_to_str(pieces) -> str:
            s = []
            for h in range(MAX_STACK):
                s += '_' if pieces[h] == -1 else str(int(pieces[h]))
            return ''.join(s) + ' '
        # rendering the whole board
        for x in range(ROWS):
            print(f"{x}: ", end="")
            for y in range(COLS):
                print(stack_to_str(self.squares[x, y, :]), end="")
            print()