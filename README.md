# Talos

#### Project Inspiration:
This project was inspired by Deepmind's Alpha Zero, a deep reinforcement learning algorithm that learns, online, how to play games by performing millions of iterations of self-play.

#### Concepts:
- Games can be represented in bits (usually)
  - For Connect Four, the bit representation could look something like a: 7x6x2 = 84 binary vector of ones and zeros. Where 7x6 represents the board's dimensions, and x3 represents the potential for a given space to have (1, 0, 0) one of your pieces in it, (0, 1, 0) no piece in it, or (0, 0, 1) one of the enemies pieces in it.
