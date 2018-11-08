# tictactoe
This code runs a user-specified number of times in training to determine a mini-max graph of the optimal moves in any state of a tic-tac-toe game. The papers referred to are listed in the code. The bot at the end of training is expected to play a sound game of tic-tac-toe using this graph.

There are two modes: Training and Playing

## Training
In Training Mode, bot creates a table of weights for all states that can follow a specific state. A state is defined as the current positions of Xs and Os on the grid specified by the coordinates. 

Note that this method uses the property of symmetry of the 3,3 grid, to reduce the number of rows in the table. Also the rate of learning decreases linearly with table size.

To learn the following method is used:
1. There is an exploration rate that partially randomizes the move of one of the bots movements against itself. This helps the bot from repeating the same actions over and over until completing training.

2. The exploration rate (e) decreases and this rate can be manipulated by the user running the program. The lowering exploration rate implies the bot has gained more mastery in the game and does not need to explore as much. (Weights are varied as a function of e, with the winner getting all their intermittent states rewarded while the losing bot getting its states penalized)

## Playing
In Play Mode, user can play with bot which has learned (to an extent) how to play tic-tac-toe. (There are some issues of local optimals being reached; *this training implementation is not perfect*)



