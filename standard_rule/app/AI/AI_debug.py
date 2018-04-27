#!/usr/bin/env python

from __future__ import print_function, division
import itertools, time, copy
import collections, random
import os, pickle
import numba
import numpy as np
import tflearn

board_size = 15
estimate_level = 3
def strategy(state):
    """ AI's strategy """

    """ Information provided to you:

    state = (board, last_move, playing, board_size)
    board = (x_stones, o_stones)
    stones is a set contains positions of one player's stones. e.g.
        x_stones = {(8,8), (8,9), (8,10), (8,11)}
    playing = 0|1, the current player's index

    Your strategy will return a position code for the next stone, e.g. (8,7)
    """

    global board_size
    board, last_move, playing, board_size = state
    strategy.playing = playing

    other_player = int(not playing)
    my_stones = board[playing]
    opponent_stones = board[other_player]

    if playing == 0: # if i'm black
        strategy.zobrist_me = strategy.zobrist_black
        strategy.zobrist_opponent = strategy.zobrist_white
    else:
        strategy.zobrist_me = strategy.zobrist_white
        strategy.zobrist_opponent = strategy.zobrist_black


    if last_move is None: # if it's the first move of the game
        best_move = (7,7)
        assert playing == 0
        strategy.hist_states = []
        strategy.started_from_beginning = True
        return (best_move[0]+1, best_move[1]+1)
    else:
        if len(my_stones) == 0:
            assert playing == 1
            strategy.started_from_beginning = True
            strategy.hist_states = []
        state = np.zeros(board_size**2, dtype=np.int32).reshape(board_size, board_size)
        strategy.zobrist_code = 0
        for i,j in my_stones:
            state[i-1,j-1] = 1
            strategy.zobrist_code ^= strategy.zobrist_me[i-1, j-1]
        for i,j in opponent_stones:
            state[i-1,j-1] = -1
            strategy.zobrist_code ^= strategy.zobrist_opponent[i-1, j-1]
        # clear the U cache
        U_stone.cache = dict()
        alpha = -2.0
        beta = 2.0
        empty_spots_left = np.sum(state==0)
        start_level = -1
        best_move, best_q = best_action_q(state, strategy.zobrist_code, empty_spots_left, last_move, alpha, beta, 1, start_level)
    if best_q != None:
        print("best_q = %.2f" % best_q)
    # return the best move
    return (best_move[0]+1, best_move[1]+1)



level_max_n = [20, 20, 15, 15, 8, 8, 6, 6, 4, 4, 4, 4, 4, 4, 4]
def best_action_q(state, zobrist_code, empty_spots_left, last_move, alpha, beta, player, level):
    "Return the optimal action for a state"
    if empty_spots_left == 0: # Board filled up, it's a tie
        return None, 0.0
    #move_interest_values = np.zeros(board_size**2, dtype=np.float32).reshape(board_size,board_size)
    move_interest_values = best_action_q.move_interest_values
    move_interest_values.fill(0) # reuse the same array
    # boost the interests of closer moves by a little bit
    # note that it might boost a taken spot, but an available spot will at least get 10 interest in find_interesting_moves()
    boost_dist = 3
    r, c = last_move
    xmin = max(0, r-boost_dist)
    xmax = min(board_size, r+boost_dist+1)
    ymin = max(0, c-boost_dist)
    ymax = min(board_size, c+boost_dist+1)
    move_interest_values[xmin:xmax, ymin:ymax] = 1.5

    is_first_move = False
    if level == -1:
       is_first_move = True
       level = 0

    verbose = is_first_move#True

    n_moves = level_max_n[level]
    interested_moves = find_interesting_moves(state, empty_spots_left, move_interest_values, player, n_moves, verbose)



    if len(interested_moves) == 1:
        current_move = interested_moves[0]
        current_move = (current_move[0], current_move[1])
        if is_first_move:
            q = None
            #Q_stone(state, zobrist_code, empty_spots_left, current_move, alpha, beta, player, estimate_level)
        else:
            q = Q_stone(state, zobrist_code, empty_spots_left, current_move, alpha, beta, player, level)
        return current_move, q

    #best_move = (-1,-1) # admit defeat if all moves have 0 win rate
    best_move = (interested_moves[0,0], interested_moves[0,1]) # continue to play even I'm losing

    if player == 1:
        max_q = -1.0
        for current_move in interested_moves:
            current_move = (current_move[0], current_move[1]) # convert into tuple
            q = Q_stone(state, zobrist_code, empty_spots_left, current_move, alpha, beta, player, level+1)
            if q > alpha: alpha = q
            if q > max_q:
                max_q = q
                best_move = current_move
            if max_q >= 1.0 or beta <= alpha:
                break
        best_q = max_q
    elif player == -1:
        min_q = 1.0
        for current_move in interested_moves:
            current_move = (current_move[0], current_move[1])
            q = Q_stone(state, zobrist_code, empty_spots_left, current_move, alpha, beta, player, level+1)
            if q < beta: beta = q
            if q < min_q:
                min_q = q
                best_move = current_move
            if q <= -1.0 or beta <= alpha:
                break
        best_q = min_q
    return best_move, best_q

def Q_stone(state, zobrist_code, empty_spots_left, current_move, alpha, beta, player, level):
    # update the state
    state[current_move] = player
    # update the zobrist code for the new state
    if player == 1:
        move_code = strategy.zobrist_me[current_move]
    else:
        move_code = strategy.zobrist_opponent[current_move]
    new_zobrist_code = zobrist_code ^ move_code

    result = U_stone(state, new_zobrist_code, empty_spots_left-1, current_move, alpha, beta, player, level)
    # revert the changes for the state
    state[current_move] = 0
    return result

def U_stone(state, zobrist_code, empty_spots_left, last_move, alpha, beta, player, level):
    try:
        return U_stone.cache[zobrist_code]
    except:
        pass
    if i_will_win(state, last_move, player):
        result = 1.0 if player == 1 else -1.0
    elif level >= estimate_level:
        result = tf_predict_u(state, zobrist_code, empty_spots_left, last_move, player)
    else:
        best_move, best_q = best_action_q(state, zobrist_code, empty_spots_left, last_move, alpha, beta, -player, level)
        result = best_q
    U_stone.cache[zobrist_code] = result
    return result

def tf_predict_u(state, zobrist_code, empty_spots_left, last_move, player):
    "Generate the best moves, use the neural network to predict U, return the max U"
    if empty_spots_left == 0: # Board filled up, it's a tie
        return 0.0
    move_interest_values = best_action_q.move_interest_values
    move_interest_values.fill(0) # reuse the same array

    next_player = -player
    n_moves = 20
    interested_moves = find_interesting_moves(state, empty_spots_left, move_interest_values, next_player, n_moves)
    # make sure we solve all the hard 4 so that there're more than one interested_moves
    next_zobrist_code = zobrist_code
    # avoid changing the original state
    next_state = state.copy() if len(interested_moves) == 1 else state
    while len(interested_moves) == 1:
        next_move = interested_moves[0]
        next_move = (next_move[0], next_move[1])
        # update the state
        next_state[next_move] = next_player
        # update the zobrist code for the new state
        if next_player == 1:
            move_code = strategy.zobrist_me[next_move]
        else:
            move_code = strategy.zobrist_opponent[next_move]
        next_zobrist_code ^= move_code
        empty_spots_left -= 1
        if empty_spots_left <= 1:
            return 0 # it's a tie
        # check if this is a win state
        if i_will_win(next_state, next_move, next_player):
            result = 1.0 if next_player == 1 else -1.0
            return result
        # check if we learned this state before
        try:
            if next_player == 1:
                return strategy.learndata[next_zobrist_code][1]
        except KeyError:
            pass
        # if we reach here, it's time to go one more step
        next_player = -next_player
        move_interest_values.fill(0)
        interested_moves = find_interesting_moves(next_state, empty_spots_left, move_interest_values, next_player, n_moves)

    # find the known moves among interested_moves
    tf_moves, move_zobrist_codes = [], [] # all unknown moves will be evaluated by tf_evaluate_max_u
    max_q = -1.0
    zobrist_map = strategy.zobrist_me if next_player == 1 else strategy.zobrist_opponent
    learndata = strategy.learndata
    for this_move in interested_moves:
        this_move = (this_move[0], this_move[1])
        this_zobrist_code = next_zobrist_code ^ zobrist_map[this_move]
        try:
            max_q = max(max_q, learndata[this_zobrist_code][1])
        except KeyError:
            try:
                max_q = max(max_q, tf_predict_u.cache[this_zobrist_code])
            except KeyError:
                tf_moves.append(this_move)
                move_zobrist_codes.append(this_zobrist_code)

    # run tensorflow to evaluate all unknown moves and find the largest q
    n_tf = len(tf_moves)
    if n_tf > 0:
        all_interest_states = tf_predict_u.all_interest_states[:n_tf] # we only need a slice of the big array
        if next_player == -1: # if next is opponent
            all_interest_states[:,:,:,0] = (next_state == -1)
            all_interest_states[:,:,:,1] = (next_state == 1)
            all_interest_states[:,:,:,2] = 1 if strategy.playing == 1 else 0 # if I'm white, next player is black
        elif next_player == 1: # if next is me
            all_interest_states[:,:,:,0] = (next_state == 1)
            all_interest_states[:,:,:,1] = (next_state == -1)
            all_interest_states[:,:,:,2] = 1 if strategy.playing == 0 else 0 # if I'm black, next is me so black
        for i,current_move in enumerate(tf_moves):
            ci, cj = current_move
            all_interest_states[i,ci,cj,0] = 1 # put current move down
        predict_y = tf_predict_u.model.predict(all_interest_states)
        predict_y = np.array(predict_y).flatten()
        #print_state(next_state)
        #for m,y in zip(tf_moves, predict_y):
        #     print(m,y)
        # store the computed y
        for zcode,y in zip(move_zobrist_codes, predict_y):
            tf_predict_u.cache[zcode] = y    
        tf_y = np.max(predict_y)
        max_q = max(max_q, tf_y)
    return max_q * next_player # if next_player is opponent, my win rate is negative of his


@numba.jit(nopython=True, nogil=True)
def find_interesting_moves(state, empty_spots_left, move_interest_values, player, n_moves, verbose=False):
    """ Look at state and find the interesing n_move moves.
    input:
    -------
    state: numpy.array board_size x board_size
    empty_spots_left: number of empty spots on the board
    player: 1 or -1, the current player
    n_moves: int, desired number of interesing moves

    output:
    -------
    interested_moves: numpy.array final_n_moves x 2
        *note : final_n_moves = 1 if limited
        *       else final_n_moves = n_moves + number of length-4 moves
        *note2: final_n_moves will not exceed empty_spots_left


    #suggested_n_moves: suggested number of moves to
    """
    force_to_block = False
    exist_will_win_move = False
    directions = ((1,1), (1,0), (0,1), (1,-1))
    final_single_move = np.zeros(2, dtype=np.int64).reshape(1,2) # for returning the single move
    for r in range(board_size):
        for c in range(board_size):
            if state[r,c] != 0: continue
            interest_value = 10 # as long as it's a valid point, this is for avoiding the taken spaces
            my_hard_4 = 0
            for dr, dc in directions:
                my_line_length = 1 # last_move
                opponent_line_length = 1
                # try to extend in the positive direction (max 5 times to check overline)
                ext_r = r
                ext_c = c
                skipped_1 = 0
                my_blocked = False
                opponent_blocked = False
                for i in range(5):
                    ext_r += dr
                    ext_c += dc
                    if ext_r < 0 or ext_r >= board_size or ext_c < 0 or ext_c >= board_size:
                        break
                    elif state[ext_r, ext_c] == player:
                        if my_blocked is True:
                            break
                        else:
                            my_line_length += 1
                            opponent_blocked = True
                    elif state[ext_r, ext_c] == -player:
                        if opponent_blocked is True:
                            break
                        else:
                            opponent_line_length += 1
                            my_blocked = True
                    elif skipped_1 is 0:
                        skipped_1 = i + 1 # allow one skip and record the position of the skip
                    else:
                        break
                # the backward counting starts at the furthest "unskipped" stone
                forward_my_open = False
                forward_opponent_open = False
                if skipped_1 == 0:
                    my_line_length_back = my_line_length
                    opponent_line_length_back = opponent_line_length
                elif skipped_1 == 1:
                    my_line_length_back = 1
                    opponent_line_length_back = 1
                    forward_my_open = True
                    forward_opponent_open = True
                else:
                    if my_blocked is False:
                        my_line_length_back = skipped_1
                        opponent_line_length_back = 1
                        forward_my_open = True
                    else:
                        my_line_length_back = 1
                        opponent_line_length_back = skipped_1
                        forward_opponent_open = True
                my_line_length_no_skip = my_line_length_back
                opponent_line_length_no_skip = opponent_line_length_back

                # backward is a little complicated, will try to extend my stones first
                ext_r = r
                ext_c = c
                skipped_2 = 0
                opponent_blocked = False
                for i in range(6-my_line_length_no_skip):
                    ext_r -= dr
                    ext_c -= dc
                    if ext_r < 0 or ext_r >= board_size or ext_c < 0 or ext_c >= board_size:
                        break
                    elif state[ext_r, ext_c] == player:
                        my_line_length_back += 1
                        opponent_blocked = True
                    elif skipped_2 is 0 and state[ext_r, ext_c] == 0:
                        skipped_2 = i + 1
                    else:
                        break
                # see if i'm winning
                if my_line_length_back == 5:
                    # if there are 5 stones in backward counting, and it's not skipped in the middle
                    if skipped_2 == 0 or skipped_2 == (6-my_line_length_no_skip):
                        # i will win with this move, I will place the stone
                        final_single_move[0,0] = r
                        final_single_move[0,1] = c
                        return final_single_move

                # extend my forward line length to check if there is hard 4
                if skipped_2 is 0:
                    my_line_length += my_line_length_back - my_line_length_no_skip
                else:
                    my_line_length += skipped_2 - 1

                backward_my_open = True if skipped_2 > 0 else False
                backward_opponent_open = False
                # then try to extend the opponent
                if opponent_blocked is True:
                    if skipped_2 == 1:
                        backward_opponent_open = True
                    skipped_2 = 0 # reset the skipped_2 here to enable the check of opponent 5 later
                else:
                    ext_r = r
                    ext_c = c
                    skipped_2 = 0
                    for i in range(6-opponent_line_length_no_skip):
                        ext_r -= dr
                        ext_c -= dc
                        if ext_r < 0 or ext_r >= board_size or ext_c < 0 or ext_c >= board_size:
                            break
                        elif state[ext_r, ext_c] == -player:
                            opponent_line_length_back += 1
                        elif skipped_2 is 0 and state[ext_r, ext_c] == 0:
                            skipped_2 = i + 1
                        else:
                            break
                    # extend opponent forward line length to check if there is hard 4
                    if skipped_2 is 0:
                        opponent_line_length += opponent_line_length_back - opponent_line_length_no_skip
                    else:
                        opponent_line_length += skipped_2 - 1
                        backward_opponent_open = True
                        # here if opponent_line_length_back == 5, skipped_2 will be 0 and this flag won't be True
                        # but it do not affect our final result, because we have to block this no matter if it's open

                # check if we have to block this
                if opponent_line_length_back == 5:
                    if (skipped_2 == 0) or (skipped_2 == 6-opponent_line_length_no_skip):
                        final_single_move[0,0] = r
                        final_single_move[0,1] = c
                        force_to_block = True
                if force_to_block is False:
                    # if I will win after this move, I won't consider other moves
                    if forward_my_open is True and my_line_length == 4:
                        my_hard_4 += 1
                    if backward_my_open is True and my_line_length_back == 4:
                        my_hard_4 += 1
                    if my_hard_4 >= 2:
                        final_single_move[0,0] = r
                        final_single_move[0,1] = c
                        exist_will_win_move = True
                if force_to_block is False and exist_will_win_move is False:
                    # compute the interest_value for other moves
                    # if any line length >= 5, it's an overline so skipped
                    if (forward_my_open is True) and (my_line_length < 5):
                        interest_value += my_line_length ** 4
                    if (backward_my_open is True) and (my_line_length_back < 5):
                        interest_value += my_line_length_back ** 4
                    if (forward_opponent_open is True) and (opponent_line_length < 5):
                        interest_value += opponent_line_length ** 4
                    if (backward_opponent_open is True) and (opponent_line_length_back < 5):
                        interest_value += opponent_line_length_back ** 4
            # after looking at all directions, record the total interest_value of this move
            move_interest_values[r, c] += interest_value
            if interest_value > 256: # one (length_4) ** 4, highly interesting move
                n_moves += 1

    # all moves have been investigated now see if we have to block first
    if force_to_block is True or exist_will_win_move is True:
        if verbose is True:
            print(final_single_move[0,0], final_single_move[0,1], "Only One")
        return final_single_move
    else:
        flattened_interest = move_interest_values.ravel()
        # The interest value > 250 means at least one length_4 or three length_3 which make it highly interesting
        #n_high_interest_moves = np.sum(flattened_interest > 266) # did it in the loop
        if n_moves > empty_spots_left:
            n_moves = empty_spots_left
        high_interest_idx = np.argsort(flattened_interest)[-n_moves:][::-1]
        interested_moves = np.empty(n_moves*2, dtype=np.int64).reshape(n_moves, 2)
        interested_moves[:,0] = high_interest_idx // board_size
        interested_moves[:,1] = high_interest_idx % board_size

        if verbose is True:
            print("There are", n_moves, "interested_moves")
            for i in range(n_moves):
                print(interested_moves[i,0],interested_moves[i,1],'  :  ', flattened_interest[high_interest_idx[i]])
        return interested_moves


@numba.jit(nopython=True,nogil=True)
def i_win(state, last_move, player):
    """ Return true if I just got 5-in-a-row with last_move """
    r, c = last_move
    # try all 4 directions, the other 4 is included
    directions = [(1,1), (1,0), (0,1), (1,-1)]
    for dr, dc in directions:
        line_length = 1 # last_move
        # try to extend in the positive direction (max 4 times)
        ext_r = r
        ext_c = c
        for _ in range(5):
            ext_r += dr
            ext_c += dc
            if ext_r < 0 or ext_r >= board_size or ext_c < 0 or ext_c >= board_size:
                break
            elif state[ext_r, ext_c] == player:
                line_length += 1
            else:
                break
        # try to extend in the opposite direction
        ext_r = r
        ext_c = c
        for _ in range(6-line_length):
            ext_r -= dr
            ext_c -= dc
            if ext_r < 0 or ext_r >= board_size or ext_c < 0 or ext_c >= board_size:
                break
            elif state[ext_r, ext_c] == player:
                line_length += 1
            else:
                break
        if line_length is 5:
            return True # 5 in a row
    return False

@numba.jit(nopython=True,nogil=True)
def i_lost(state, player):
    for r in range(board_size):
        for c in range(board_size):
            if state[r,c] == 0 and i_win(state, (r,c), -player):
                return True
    return False

@numba.jit(nopython=True,nogil=True)
def i_will_win(state, last_move, player):
    """ Return true if I will win next step if the opponent don't have 4-in-a-row.
    Winning Conditions:
        1. 5 in a row.
        2. 4 in a row with both end open. (free 4)
        3. 4 in a row with one missing stone x 2 (hard 4 x 2)
     """
    r, c = last_move
    # try all 4 directions, the other 4 is equivalent
    directions = [(1,1), (1,0), (0,1), (1,-1)]
    n_hard_4 = 0 # number of hard 4s found
    for dr, dc in directions:
        line_length = 1 # last_move
        # try to extend in the positive direction (max 5 times to check overline)
        ext_r = r
        ext_c = c
        skipped_1 = 0
        for i in range(5):
            ext_r += dr
            ext_c += dc
            if ext_r < 0 or ext_r >= board_size or ext_c < 0 or ext_c >= board_size:
                break
            elif state[ext_r, ext_c] == player:
                line_length += 1
            elif skipped_1 is 0 and state[ext_r, ext_c] == 0:
                skipped_1 = i+1 # allow one skip and record the position of the skip
            else:
                break
        # try to extend in the opposite direction
        ext_r = r
        ext_c = c
        skipped_2 = 0
        # the backward counting starts at the furthest "unskipped" stone
        if skipped_1 is not 0:
            line_length_back = skipped_1
        else:
            line_length_back = line_length
        line_length_no_skip = line_length_back
        for i in range(6-line_length_back):
            ext_r -= dr
            ext_c -= dc
            if ext_r < 0 or ext_r >= board_size or ext_c < 0 or ext_c >= board_size:
                break
            elif state[ext_r, ext_c] == player:
                line_length_back += 1
            elif skipped_2 is 0 and state[ext_r, ext_c] == 0:
                skipped_2 = i + 1
            else:
                break

        if line_length_back == 6:
            # we found 6 stones in a row, this is overline, skip this entire line
            continue
        elif line_length_back == 5:
            if (skipped_2 == 0) or (skipped_2 == (6-line_length_no_skip)):
                # we found 5 stones in a row, because the backward counting is not skipped in the middle
                return True
                # else there is an empty spot in the middle of 6 stones, it's not a hard 4 any more
        elif line_length_back == 4:
            # here we have only 4 stones, if skipped in back count, it's a hard 4
            if skipped_2 is not 0:
                n_hard_4 += 1 # backward hard 4
                if n_hard_4 == 2:
                    return True # two hard 4
        # here we check if there's a hard 4 in the forward direction
        # extend the forward line to the furthest "unskipped" stone
        if skipped_2 == 0:
            line_length += line_length_back - line_length_no_skip
        else:
            line_length += skipped_2 - 1
        # hard 4 only if forward length is 4, if forward reaches 5 or more, it's going to be overline
        if line_length == 4 and skipped_1 is not 0:
            n_hard_4 += 1 # forward hard 4
            if n_hard_4 == 2:
                return True # two hard 4 or free 4
    return False



def initialize():
    # initialize zobrist for u caching
    if not hasattr(strategy, 'zobrist_me'):
        np.random.seed(20180104) # use the same random matrix for storing
        strategy.zobrist_black = np.random.randint(np.iinfo(np.int64).max, size=board_size**2).reshape(board_size,board_size)
        strategy.zobrist_white = np.random.randint(np.iinfo(np.int64).max, size=board_size**2).reshape(board_size,board_size)
        #strategy.zobrist_code = np.random.randint(np.iinfo(np.int64).max)
        # reset the random seed to random for other functions
        np.random.seed()

    if not hasattr(best_action_q, 'move_interest_values'):
        best_action_q.move_interest_values = np.zeros(board_size**2, dtype=np.float32).reshape(board_size,board_size)

    if not hasattr(strategy, 'learndata'):
        if os.path.isfile('strategy.learndata'):
            strategy.learndata = pickle.load(open('strategy.learndata','rb'))
            print("strategy.learndata found, loaded %d data" % len(strategy.learndata))
        else:
            strategy.learndata = dict()
    strategy.started_from_beginning = False

    if not hasattr(tf_predict_u, 'all_interest_states'):
        tf_predict_u.all_interest_states = np.zeros(board_size**4 * 3, dtype=np.int8).reshape(board_size**2, board_size, board_size, 3)

    if not hasattr(tf_predict_u, 'cache'):
        if os.path.isfile('tf_predict_u.cache'):
            tf_predict_u.cache = pickle.load(open("tf_predict_u.cache",'rb'))
            print("tf_predict_u.cache found, loaded %d cache" % len(tf_predict_u.cache))
        else:
            tf_predict_u.cache = dict()

    if not hasattr(tf_predict_u, 'model'):
        tflearn.init_graph(num_cores=4, gpu_memory_fraction=0.3)
        import construct_dnn
        model = construct_dnn.construct_dnn()
        path = os.path.realpath(__file__)
        folder = os.path.dirname(path)
        model.load(os.path.join(folder,'tf_model'))
        tf_predict_u.model = model

def board_show(stones):
    if isinstance(stones, np.ndarray):
        stones = {(s1,s2) for s1, s2 in stones}
    print(' '*4 + ' '.join([chr(97+i) for i in xrange(board_size)]))
    print (' '*3 + '='*(2*board_size))
    for x in xrange(1, board_size+1):
        row = ['%2s|'%x]
        for y in xrange(1, board_size+1):
            if (x-1,y-1) in stones:
                c = 'x'
            else:
                c = '-'
            row.append(c)
        print (' '.join(row))

def print_state(state):
    assert isinstance(state, np.ndarray)
    print(' '*4 + ' '.join([chr(97+i) for i in xrange(board_size)]))
    print (' '*3 + '='*(2*board_size))
    for x in xrange(1, board_size+1):
        row = ['%2s|'%x]
        for y in xrange(1, board_size+1):
            if state[x-1,y-1] == 1:
                c = 'o'
            elif state[x-1,y-1] == -1:
                c = 'x'
            else:
                c = '-'
            row.append(c)
        print (' '.join(row))


def test():
    initialize()
    state = np.zeros(board_size**2, dtype=np.int8).reshape(board_size, board_size)
    state[(9,10,11),(11,12,13)] = 1
    state[(6,8,9,11),(7,9,10,12)] = -1
    lastmove = (8,9)
    print_state(state)
    find_interesting_moves(state, 210, best_action_q.move_interest_values, player=1, n_moves=20, verbose=True)


def draw_state(state):
    board_size = 15
    print(' '*4 + ' '.join([chr(97+i) for i in xrange(board_size)]))
    print (' '*3 + '='*(2*board_size))
    me = state[0,0,2]
    for x in xrange(1, board_size+1):
        row = ['%2s|'%x]
        for y in xrange(1, board_size+1):
            if state[x-1,y-1,0] == 1:
                c = 'x' if me == 1 else 'o'
            elif state[x-1,y-1,1] == 1:
                c = 'o' if me == 1 else 'x'
            else:
                c = '-'
            row.append(c)
        print (' '.join(row))

if __name__ == '__main__':
    #import pickle
    #state = pickle.load(open('debug.state','rb'))
    #board, last_move, playing, board_size = state
    #player_stones = board[playing]
    #other = int(not playing)
    #ai_stones = board[other]
    #player_move = (8,9)
    #player_stones.add(player_move)
    #state = (player_stones, ai_stones), player_move, other, board_size
    #strategy(state)
    test()
