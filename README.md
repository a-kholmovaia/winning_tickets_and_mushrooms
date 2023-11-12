# Iterative magnitude pruning from scratch on mushroom data
I reimplemented the code from the paper [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635) for the [mushroom data](https://archive.ics.uci.edu/dataset/73/mushroom).

## Iterative magnitude pruning 
The algorithm can be described as follows:
- randomly initialise a network $W_i$
- train the network for $n$ epochs
- prune p% of the trained weights with the smallest magnitude and create a mask m
- reinitialise unpruned weights with $W_i$ ('rewind' the weights)
- repeat the training, pruning and reinitialising steps for k times
- get the winning ticket
### Key takeaways from the paper
  - By iterative magnitude pruning we can find subnetworks that are only 10-20% of the size of the original network (without any performance loss).
  - But the structure of the the ticket (final  mask) isn't enough. If we were to reinitialise the ticket, retaining the structure, but with different weights, the ticket wouldn't perform as well as with the initial weights $W_i$. 

