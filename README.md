## AlphaZero

A clean implementation of the AlphaZero method as described in [Mastering the Game of Go without Human Knowledge](https://deepmind.com/research/publications/2019/mastering-game-go-without-human-knowledge) that can be applied to an arbitrary full-information two player game with an example implementation for [Connect Four](https://en.wikipedia.org/wiki/Connect_Four), based on PyTorch.

* `train.py` contains the main training code with the opportunity to customize hyperparameters, the game being played, the model and the ability to save and load models and training replay buffers.
* `game.py` contains the implementation of the game logic.
* `mcts.py` contains the main logic for Monte-Carlo Tree Search as described in the paper.
* `model.py` contains the model definition.
* `play.py` allows a human to play against selected previously stored models.
* `tournament.py` pits several models against each other and prints out a final score board.
* `utils.py` contains some utility functionality.
* `visualize.py` contains code to visualize game states.

&nbsp;

![Visualization of example gameplay](example.gif)