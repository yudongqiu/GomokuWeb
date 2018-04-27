# coding: utf-8
import numpy as np
#st = np.loadtxt('st.log')
state = np.zeros((15,15), dtype=np.int8)
#st[8,6:9] = -1
#from AI_debug import find_interesting_moves
#move_interest_values = np.zeros((15,15), dtype=int)
#find_interesting_moves(st, 200, move_interest_values, -1, 20, True)
from AI_debug import i_win, print_state

state[8,1] = -1
state[(7,6,5,4,3,2),(2,3,4,5,6,7)] = 1
print_state(state)
print i_win(state, (3,6), 1)
