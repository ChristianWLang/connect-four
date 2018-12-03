# Connect-Four AI

#### Project Inspiration:
This project was inspired by Deepmind's Alpha Zero, a deep reinforcement learning algorithm that learns, online, how to play games by performing millions of iterations of self-play.

#### Concepts:
- Games can be represented in bits (usually)
  - For Connect Four, the bit representation could look something like a: 7x6x2 = 84 binary vector of ones and zeros. Where 7x6 represents the board's dimensions, and x2 represents the potential for a given space to have one of your pieces in it ([1, 0]), no piece in it ([0, 0]), or one of the opponent's pieces in it ([0, 1])
- The Most effective way (at current time) to traverse a zero-sum, fully deterministic, perfect information game is by tree search (given that the game is not easily brute force-able, which, ironically, connect four is)

#### Usage:
```
$ pip3 install -r requirements.txt
# To train the model
$ python3 -m src.main
```

#### Requirements:
As of writing this, python >= 3.7 is incompatible with this module.

#### TODO:
Figure out why search results aren't being reused.
