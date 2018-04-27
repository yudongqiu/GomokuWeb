# GomokuWeb
A web version of the gomoku game with AI I wrote

Dependencies: Python2, Numpy, Numba, Flask (All comes with Anaconda2)

To run the game server, enter command: ```python run.py```

Then open the website http://127.0.0.1:5000/ in your browser to play!

The player name box determines who plays first, i.e., to let the AI play first, simply put "AI" in "Black" and "You" in "White".

The AI Level box determines the number of moves AI will predict. Currently the default is 7 so it will compute a step in less than 1s. (But it's actually not easy.) If you are good at this game and want a challenge, give AI Level 10 a try. In my test Level > 11 will be pretty slow. 

To let 2 AI play against each other, put "AI" and "AI2" in the "Black" and "White" box.

An DNN powered AI is added. To play against it, you need to install tensorflow and tflearn first, then enter "AI_tf" in the name textbox.

It is suggested to choose AI Level 3 or smaller because the DNN evaluation is slow. But you will be surprised by the level of skills. :P

If you have a new AI script, please name it starting from "AI", e.g. "AI_new.py", and put corresponding name in the box.

Please Enjoy!

Note: Special thanks to @lihongxun945 for the beautiful design of the web page!
