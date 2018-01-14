# coding: utf-8
import numpy as np
#st = np.loadtxt('st.log')
st = np.zeros((15,15), dtype=np.int8)
st[8,6:9] = -1
from AI_debug import find_interesting_moves
move_interest_values = np.zeros((15,15), dtype=int)
find_interesting_moves(st, 200, move_interest_values, -1, 20, True)
