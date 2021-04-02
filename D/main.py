import os
from PIL import Image
import numpy as np
from copy import deepcopy


def read_image(dir_path):
    image = None
    for file in os.listdir(dir_path):
        if file.endswith('.png'):
            image = Image.open(os.path.join(dir_path, file))
    return image


def find_corner_coordinates(img_mat):
    n, m, _ = img_mat.shape
    for i in range(n):
        for j in range(m):
            if img_mat[i, j, 0] != 0 and img_mat[i, j, 1] != 0 and img_mat[i, j, 0] != 0:
                return i, j


def calculate_board_dim(img_mat, start_pos):
    """
    Calculates the length of one side of the board (all info we need to find specific tiles)
    """
    x, y = start_pos
    dim = 0  # size of one side of the chess board
    while img_mat[x, y, 0] != 0 and img_mat[x, y, 1] != 0 and img_mat[x, y, 0] != 0:
        dim += 1
        x += 1  # move one pixel to the right until we hit black one
    return dim


dict_pieces = dict()  # pieces with black-tile background


def create_pieces_library(path, dim):
    """
    Creates two dictionaries for all pieces on both type of color, one for black tiles
    and one for white
    """
    global dict_pieces

    pieces_path_black = path + r"\pieces\black"
    pieces_path_white = path + r"\pieces\white"
    for img in os.listdir(pieces_path_black):
        piece = Image.open(os.path.join(pieces_path_black, img))
        background = Image.new("1", piece.size, 183)
        background.paste(piece, (0, 0), piece.convert("RGBA"))
        background = background.convert('L')
        piece_image = background.resize((dim, dim))

        if img.endswith('bishop.png'):
            dict_pieces['bishop'] = piece_image
        elif img.endswith('king.png'):
            dict_pieces['king'] = piece_image
        elif img.endswith('knight.png'):
            dict_pieces['night'] = piece_image
        elif img.endswith('pawn.png'):
            dict_pieces['pawn'] = piece_image
        elif img.endswith('queen.png'):
            dict_pieces['queen'] = piece_image
        elif img.endswith('rook.png'):
            dict_pieces['rook'] = piece_image

    for img in os.listdir(pieces_path_white):
        piece = Image.open(os.path.join(pieces_path_white, img))
        background = Image.new("1", piece.size, 183)
        background.paste(piece, (0, 0), piece.convert("RGBA"))
        background = background.convert('L')
        piece_image = background.resize((dim, dim))

        if img.endswith('bishop.png'):
            dict_pieces['Bishop'] = piece_image
        elif img.endswith('king.png'):
            dict_pieces['King'] = piece_image
        elif img.endswith('knight.png'):
            dict_pieces['Night'] = piece_image
        elif img.endswith('pawn.png'):
            dict_pieces['Pawn'] = piece_image
        elif img.endswith('queen.png'):
            dict_pieces['Queen'] = piece_image
        elif img.endswith('rook.png'):
            dict_pieces['Rook'] = piece_image


def cost_function(arr1, arr2):
    res = np.sum(np.abs(arr1 - arr2))
    return res


def find_best_fit(field_arr):
    """
    Find the item in the dictionary that is the most similar to the field_arr parameter based on the
    chosen cost function
    """
    dictionary = dict_pieces
    min_value = float('inf')
    piece = ""
    for key, value in dictionary.items():
        piece_arr = np.array(value, dtype=int)
        cost = cost_function(field_arr, piece_arr)
        if cost < min_value:
            min_value = cost
            piece = key
    return piece


def detect_piece(board, start, idx, idy, dim_f):
    """
    If it is established that a field has a piece, it returns its type and color
    Board has to be grayscale
    """
    global dict_pieces
    # left upper corner of the field with coordinates idx, idy
    upper_left = (start[0] + idx * dim_f, start[1] + idy * dim_f)
    lower_right = (upper_left[0] + dim_f, upper_left[1] + dim_f)
    field_crop = board.crop((upper_left[1], upper_left[0], lower_right[1], lower_right[0]))

    field_arr = np.array(field_crop, dtype=int)
    if field_arr[dim_f // 2 - 1, dim_f // 2 - 1] != 220 and field_arr[dim_f // 2 - 1, dim_f // 2 - 1] != 145:
        return find_best_fit(field_arr)[0]
    else:
        return None


def reproduce_board(board, start, dim_b):
    """
    Creates 8x8 matrix representing current chess board position on the image
    """
    board_matrix = [[None for i in range(8)] for j in range(8)]
    dim_f = dim_b // 8
    for i in range(8):
        for j in range(8):
            board_matrix[i][j] = detect_piece(board, start, i, j, dim_f)
    return board_matrix


def fen_notation(board, start, dim_b):
    board_matrix = reproduce_board(board, start, dim_b)
    acc = 0
    for i in range(8):
        for j in range(8):
            if board_matrix[i][j] is None:
                acc += 1
            elif acc:
                print("{0}{1}".format(acc, board_matrix[i][j]), end="")
                acc = 0
            else:
                print(board_matrix[i][j], end="")
                acc = 0
            if j == 7:
                if i != 7:
                    if acc:
                        print("{}/".format(acc), end="")
                        acc = 0
                    else:
                        print("/", end="")
                elif acc:
                    print(acc)
                else:
                    print()
    return board_matrix


def king_attack(x, y, bit_board):
    # possible coordinates
    dx = [1, -1, 0, 0, 1, -1, 1, -1]
    dy = [0, 0, 1, -1, 1, -1, -1, 1]
    for i in range(len(dx)):
        t_x = x + dx[i]
        t_y = y + dy[i]
        if 0 <= t_x < 8 and 0 <= t_y < 8:
            bit_board[t_x][t_y].append((x, y))


def pawn_attack(color, x, y, bit_board):
    dy = [-1, 1]
    if color:
        # black
        dx = [1, 1]
    else:
        # white
        dx = [-1, -1]
    for i in range(len(dx)):
        t_x = x + dx[i]
        t_y = y + dy[i]
        if 0 <= t_x < 8 and 0 <= t_y < 8:
            bit_board[t_x][t_y].append((x, y))


def bishop_attack(x, y, board, bit_board):
    dx = [1, 1, -1, -1]
    dy = [-1, 1, -1, 1]
    for i in range(len(dx)):
        curr_x = x + dx[i]
        curr_y = y + dy[i]
        while 0 <= curr_x < 8 and 0 <= curr_y < 8 and board[curr_x][curr_y] is None:
            bit_board[curr_x][curr_y].append((x, y))
            curr_x += dx[i]
            curr_y += dy[i]
        if 0 <= curr_x < 8 and 0 <= curr_y < 8:
            bit_board[curr_x][curr_y].append((x, y))


def knight_attack(x, y, bit_board):
    # possible coordinates
    dx = [2, 1, -1, -2, -2, -1, 1, 2]
    dy = [1, 2, 2, 1, -1, -2, -2, -1]

    for i in range(len(dx)):
        t_x = x + dx[i]
        t_y = y + dy[i]
        if 0 <= t_x < 8 and 0 <= t_y < 8:
            bit_board[t_x][t_y].append((x, y))


def rook_attack(x, y, board, bit_board):
    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1]
    for i in range(len(dx)):
        curr_x = x + dx[i]
        curr_y = y + dy[i]
        while 0 <= curr_x < 8 and 0 <= curr_y < 8 and board[curr_x][curr_y] is None:
            bit_board[curr_x][curr_y].append((x, y))
            curr_x += dx[i]
            curr_y += dy[i]
        if 0 <= curr_x < 8 and 0 <= curr_y < 8:
            bit_board[curr_x][curr_y].append((x, y))


def queen_attack(x, y, board, bit_board):
    rook_attack(x, y, board, bit_board)
    bishop_attack(x, y, board, bit_board)


def check_for_checks(board):
    attack_board_white = [[[] for j in range(8)] for i in range(8)]
    attack_board_black = [[[] for j in range(8)] for i in range(8)]
    white_king = tuple()
    black_king = tuple()

    for i in range(8):
        for j in range(8):
            if board[i][j] is None:
                continue
            elif board[i][j] == 'p':
                pawn_attack(1, i, j, attack_board_black)
            elif board[i][j] == 'P':
                pawn_attack(0, i, j, attack_board_white)
            elif board[i][j] == 'k':
                king_attack(i, j, attack_board_black)
                black_king = (i, j)
            elif board[i][j] == 'K':
                king_attack(i, j, attack_board_white)
                white_king = (i, j)
            elif board[i][j] == 'n':
                knight_attack(i, j, attack_board_black)
            elif board[i][j] == 'N':
                knight_attack(i, j, attack_board_white)
            elif board[i][j] == 'b':
                bishop_attack(i, j, board, attack_board_black)
            elif board[i][j] == 'B':
                bishop_attack(i, j, board, attack_board_white)
            elif board[i][j] == 'r':
                rook_attack(i, j, board, attack_board_black)
            elif board[i][j] == 'R':
                rook_attack(i, j, board, attack_board_white)
            elif board[i][j] == 'q':
                queen_attack(i, j, board, attack_board_black)
            elif board[i][j] == 'Q':
                queen_attack(i, j, board, attack_board_white)
    defending_side = 0
    if len(attack_board_white[black_king[0]][black_king[1]]):
        print("W")
        defending_side = -1
    elif len(attack_board_black[white_king[0]][white_king[1]]):
        print("B")
        defending_side = 1
    else:
        print("-")
    return white_king, attack_board_white, black_king, attack_board_black, defending_side


def check_king_moves(king_pos, piece_board, attacking_board, defending_side):
    # possible moves
    dx = [1, -1, 0, 0, 1, -1, 1, -1]
    dy = [0, 0, 1, -1, 1, -1, -1, 1]
    x, y = king_pos

    for i in range(len(dx)):
        t_x = x + dx[i]
        t_y = y + dy[i]
        if 0 <= t_x < 8 and 0 <= t_y < 8:
            if len(attacking_board[t_x][t_y]) == 0:
                if defending_side == -1:  # black side
                    if piece_board[t_x][t_y] is None or piece_board[t_x][t_y].isupper():
                        piece_board[x][y] = None
                        piece_holder = piece_board[t_x][t_y]
                        tmp_board = np.zeros((8, 8))
                        escaped = True
                        for attacker_pos in attacking_board[x][y]:
                            attacker_x, attacker_y = attacker_pos
                            if piece_board[attacker_x][attacker_y] == 'R':
                                rook_attack_bit(attacker_x, attacker_y, piece_board, tmp_board)
                            elif piece_board[attacker_x][attacker_y] == 'B':
                                bishop_attack_bit(attacker_x, attacker_y, piece_board, tmp_board)
                            elif piece_board[attacker_x][attacker_y] == 'Q':
                                rook_attack_bit(attacker_x, attacker_y, piece_board, tmp_board)
                                bishop_attack_bit(attacker_x, attacker_y, piece_board, tmp_board)
                            if tmp_board[t_x, t_y] == 1:
                                escaped = False
                                break
                        # reverse the changes
                        piece_board[x][y] = 'k'
                        piece_board[t_x][t_y] = piece_holder
                        if escaped:
                            return False
                else:  # white side
                    if piece_board[t_x][t_y] is None or piece_board[t_x][t_y].islower():
                        piece_board[x][y] = None
                        piece_holder = piece_board[t_x][t_y]
                        tmp_board = np.zeros((8, 8))
                        escaped = True
                        for attacker_pos in attacking_board[x][y]:
                            attacker_x, attacker_y = attacker_pos
                            if piece_board[attacker_x][attacker_y] == 'r':
                                rook_attack_bit(attacker_x, attacker_y, piece_board, tmp_board)
                            elif piece_board[attacker_x][attacker_y] == 'b':
                                bishop_attack_bit(attacker_x, attacker_y, piece_board, tmp_board)
                            elif piece_board[attacker_x][attacker_y] == 'q':
                                rook_attack_bit(attacker_x, attacker_y, piece_board, tmp_board)
                                bishop_attack_bit(attacker_x, attacker_y, piece_board, tmp_board)
                            if tmp_board[t_x, t_y] == 1:
                                escaped = False
                                break
                        # reverse the changes
                        piece_board[x][y] = 'K'
                        piece_board[t_x][t_y] = piece_holder
                        if escaped:
                            return False
    return True


def bishop_attack_bit(x, y, board, bit_board):  # same as bishop_attack, only there to improve performance
    dx = [1, 1, -1, -1]
    dy = [-1, 1, -1, 1]
    for i in range(len(dx)):
        curr_x = x + dx[i]
        curr_y = y + dy[i]
        while 0 <= curr_x < 8 and 0 <= curr_y < 8 and board[curr_x][curr_y] is None:
            bit_board[curr_x, curr_y] = 1
            curr_x += dx[i]
            curr_y += dy[i]
        if 0 <= curr_x < 8 and 0 <= curr_y < 8:
            bit_board[curr_x, curr_y] = 1


def rook_attack_bit(x, y, board, bit_board):  # same as rook_attack, only there to improve performance
    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1]
    for i in range(len(dx)):
        curr_x = x + dx[i]
        curr_y = y + dy[i]
        while 0 <= curr_x < 8 and 0 <= curr_y < 8 and board[curr_x][curr_y] is None:
            bit_board[curr_x, curr_y] = 1
            curr_x += dx[i]
            curr_y += dy[i]
        if 0 <= curr_x < 8 and 0 <= curr_y < 8:
            bit_board[curr_x, curr_y] = 1


def rook_attack_id(x, y, board, king_x, king_y):
    """
    Function for determining fields that can be blockaded to stop the check of the king for a rook
    """
    indices = []
    if king_x == x and king_y < y:
        dx = 0
        dy = -1
    elif king_x == x and king_y > y:
        dx = 0
        dy = 1
    elif king_x < x and king_y == y:
        dx = -1
        dy = 0
    elif king_x > x and king_y == y:
        dx = 1
        dy = 0
    else:
        return indices
    curr_x = x + dx
    curr_y = y + dy
    while 0 <= curr_x < 8 and 0 <= curr_y < 8 and board[curr_x][curr_y] is None:
        indices.append((curr_x, curr_y))
        curr_x += dx
        curr_y += dy
    return indices


def bishop_attack_id(x, y, board, king_x, king_y):
    """
    Function for determining fields that can be blockaded to stop the check of the king for a bishop
    """
    indices = []
    if king_x > x and king_y > y:
        dx = 1
        dy = 1
    elif king_x > x and king_y < y:
        dx = 1
        dy = -1
    elif king_x < x and king_y > y:
        dx = -1
        dy = 1
    elif king_x < x and king_y < y:
        dx = -1
        dy = -1
    else:
        return indices
    curr_x = x + dx
    curr_y = y + dy
    while 0 <= curr_x < 8 and 0 <= curr_y < 8 and board[curr_x][curr_y] is None:
        indices.append((curr_x, curr_y))
        curr_x += dx
        curr_y += dy
    return indices


def check_captures(king_pos, piece_board, attacking_board, defending_board):
    # coordinates of the attacking piece
    attacker_x, attacker_y = attacking_board[king_pos[0]][king_pos[1]][0]

    for pos in defending_board[attacker_x][attacker_y]:
        defender_x, defender_y = pos
        # already checked if the attacked king has legal moves
        if piece_board[defender_x][defender_y] == 'k' or piece_board[defender_x][defender_y] == 'K':
            continue
        defender_type = piece_board[defender_x][defender_y]
        piece_board[defender_x][defender_y] = None
        curr_type = piece_board[attacker_x][attacker_y]
        piece_board[attacker_x][attacker_y] = defender_type

        # we check if defending piece is pinned to the defense of the king
        pinned = False
        tmp_board = np.zeros((8, 8))
        for potential_attacker in attacking_board[defender_x][defender_y]:
            if potential_attacker == (attacker_x, attacker_y):  # it will be captured by the attacked piece
                continue
            potential_attacker_type = piece_board[potential_attacker[0]][potential_attacker[1]]
            potential_attacker_x, potential_attacker_y = potential_attacker
            if potential_attacker_type == 'r' or potential_attacker_type == 'R':
                rook_attack_bit(potential_attacker_x, potential_attacker_y, piece_board, tmp_board)
            elif potential_attacker_type == 'b' or potential_attacker_type == 'B':
                bishop_attack_bit(potential_attacker_x, potential_attacker_y, piece_board, tmp_board)
            elif potential_attacker_type == 'q' or potential_attacker_type == 'Q':
                rook_attack_bit(potential_attacker_x, potential_attacker_y, piece_board, tmp_board)
                bishop_attack_bit(potential_attacker_x, potential_attacker_y, piece_board, tmp_board)
            else:
                continue
            if tmp_board[king_pos[0], king_pos[1]] == 1:
                pinned = True
                break
        # backtrack to the starting board position
        piece_board[defender_x][defender_y] = defender_type
        piece_board[attacker_x][attacker_y] = curr_type
        if not pinned:
            return False
    return True


def check_blockades(king_pos, piece_board, attacking_board, defending_board):
    # coordinates of the attacking piece
    attacker_x, attacker_y = attacking_board[king_pos[0]][king_pos[1]][0]
    attacker_type = piece_board[attacker_x][attacker_y]
    king_x, king_y = king_pos

    potential_fields = []
    if attacker_type == 'r' or attacker_type == 'R':
        potential_fields.extend(rook_attack_id(attacker_x, attacker_y, piece_board, king_x, king_y))
    elif attacker_type == 'b' or attacker_type == 'B':
        potential_fields.extend(bishop_attack_id(attacker_x, attacker_y, piece_board, king_x, king_y))
    elif attacker_type == 'q' or attacker_type == 'Q':
        potential_fields.extend(rook_attack_id(attacker_x, attacker_y, piece_board, king_x, king_y))
        potential_fields.extend(bishop_attack_id(attacker_x, attacker_y, piece_board, king_x, king_y))
    else:
        # only ranged pieces can be blocked
        return True
    # iterate through potential fields
    for field in potential_fields:
        curr_x, curr_y = field

        for pos in defending_board[curr_x][curr_y]:
            defender_x, defender_y = pos
            # already checked if the attacked king has legal moves
            if piece_board[defender_x][defender_y] == 'k' or piece_board[defender_x][defender_y] == 'K':
                continue
            defender_type = piece_board[defender_x][defender_y]
            piece_board[defender_x][defender_y] = None
            curr_type = piece_board[curr_x][curr_y]
            piece_board[curr_x][curr_y] = defender_type

            # we check if defending piece is pinned to the defense of the king
            pinned = False
            tmp_board = np.zeros((8, 8))
            for potential_attacker in attacking_board[defender_x][defender_y]:
                if potential_attacker == (curr_x, curr_y):  # it will be captured by the attacked piece
                    continue
                potential_attacker_type = piece_board[potential_attacker[0]][potential_attacker[1]]
                potential_attacker_x, potential_attacker_y = potential_attacker
                if potential_attacker_type == 'r' or potential_attacker_type == 'R':
                    rook_attack_bit(potential_attacker_x, potential_attacker_y, piece_board, tmp_board)
                elif potential_attacker_type == 'b' or potential_attacker_type == 'B':
                    bishop_attack_bit(potential_attacker_x, potential_attacker_y, piece_board, tmp_board)
                elif potential_attacker_type == 'q' or potential_attacker_type == 'Q':
                    rook_attack_bit(potential_attacker_x, potential_attacker_y, piece_board, tmp_board)
                    bishop_attack_bit(potential_attacker_x, potential_attacker_y, piece_board, tmp_board)
                else:
                    continue
                if tmp_board[king_pos[0], king_pos[1]] == 1:
                    pinned = True
                    break
            # backtrack to the starting board position
            piece_board[defender_x][defender_y] = defender_type
            piece_board[curr_x][curr_y] = curr_type
            if not pinned:
                return False
    return True


def reset_pawn_help(x, y, attack_board, pieces_board, color):
    dy = [-1, 1]
    if color:
        # black
        dx = [1, 1]
    else:
        # white
        dx = [-1, -1]
    if 0 <= x + dx[0] < 8 and 0 <= y + dy[0] < 8:
        attack_board[x + dx[0]][y + dy[0]].remove((x, y))
    if 0 <= x + dx[1] < 8 and 0 <= y + dy[1] < 8:
        attack_board[x + dx[1]][y + dy[1]].remove((x, y))
    if 0 <= x + dx[0] < 8 and pieces_board[x + dx[0]][y] is None:
        attack_board[x + dx[0]][y].append((x, y))
    # if pawns are in their starting position, they can move two fields forward
    if color == 1 and x == 1 and pieces_board[2][y] is None and pieces_board[3][y] is None:
        attack_board[3][y].append((x, y))
    if color == 0 and x == 6 and pieces_board[5][y] is None and pieces_board[4][y] is None:
        attack_board[4][y].append((x, y))


def reset_pawn_attack(pieces_board, attack_board, color):
    """
    Pawns capture and move differently, so it must be changed for the next phase
    """
    for i in range(8):
        for j in range(8):
            if pieces_board[i][j] == 'p' and color == 1 or pieces_board[i][j] == 'P' and color == 0:
                reset_pawn_help(i, j, attack_board, pieces_board, color)
    return attack_board


def is_checkmate(white_king, black_king, piece_board, attack_board_white, attack_board_black, defending_side):
    """
    Determines if checked king is in checkmate or not. When you are under check, you have three options:
    move the king, capture the attacking piece or blockade its path to the king (if it is a range piece).
    If a king is attacked twice, then he can only try to move to avoid checkmate.
    """
    if defending_side == -1:
        defending_board_capture = attack_board_black
        attacking_board, defending_board_blockade = attack_board_white, reset_pawn_attack(piece_board,
                                                                                          deepcopy(attack_board_black),
                                                                                          1)
        king_pos = black_king
    elif defending_side == 1:
        defending_board_capture = attack_board_white
        attacking_board, defending_board_blockade = attack_board_black, reset_pawn_attack(piece_board,
                                                                                          deepcopy(attack_board_white),
                                                                                          0)
        king_pos = white_king
    else:
        print(0)
        return False
    checkmate = True
    checkmate = check_king_moves(king_pos, piece_board, attacking_board, defending_side)
    if not checkmate:
        print(0)
        return False
    if len(attacking_board[king_pos[0]][king_pos[1]]) > 1:  # double-check on the king requires him to move
        print(1)
        return True

    checkmate = check_captures(king_pos, piece_board, attacking_board, defending_board_capture)
    if not checkmate:
        print(0)
        return False
    checkmate = check_blockades(king_pos, piece_board, attacking_board, defending_board_blockade)
    if not checkmate:
        print(0)
        return False
    else:
        print(1)
        return True


def main():
    image_path = input()

    board_image = read_image(image_path)
    img_mat = np.array(board_image)
    left_corner = find_corner_coordinates(img_mat)
    board_dim = calculate_board_dim(img_mat, left_corner)
    print("{0},{1}".format(left_corner[0], left_corner[1]))

    create_pieces_library(image_path, board_dim // 8)
    board_gray = board_image.convert('L')
    piece_board = fen_notation(board_gray, left_corner, board_dim)

    white_king, attack_board_white, black_king, attack_board_black, defending_side = check_for_checks(piece_board)

    is_checkmate(white_king, black_king, piece_board, attack_board_white, attack_board_black, defending_side)


if __name__ == "__main__":
    main()
