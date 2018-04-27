#!/usr/bin/env python
import sys, os, argparse
parser = argparse.ArgumentParser("Play Gomoku Game in a Web Browser")
parser.add_argument('-f', '--free_style', action='store_true', help='play freestyle gomoku (allow 6 stones)')
args = parser.parse_args()

if args.free_style:
    path = os.path.join(sys.path[0], 'free_style')
else:
    path = os.path.join(sys.path[0], 'standard_rule')
sys.path.append(path)
sys.path.append(os.path.join(path,'app'))
sys.path.append(os.path.join(path,'app','AI'))

from app import app
app.run(debug=True)
