# Imitation_Learning

Work in progress, deadline 19 january 2018.

Class project for the course Reinforcement Learning of the MVA 2017-2018

The goal is to train an agent to play Enduro (https://gym.openai.com/envs/Enduro-v0/) using a reinforcement learning scheme reduced to no-regret learning. The implemented article is :

* <a href="http://proceedings.mlr.press/v15/ross11a/ross11a.pdf">A Reduction of Imitation Learning and Structured Prediction
to No-Regret Online Learning</a>

To implement the algorithm we use python and the open AI gym environment. 

**To do**:
- [ ] Build Neural net
- [ ] Data set Generation
- [ ] Build machine interface

Repo architecture :
- <span style="color:green"> Algorithms.py</span> (SMILE DAGGER)
- <span style="color:red"> Data</span>
    - game
        - images (.npy)
        - actions (.npy)
        - states_list (.txt)
        - states (.npy)
- <span style="color:green"> main.py</span>
- <span style="color:red"> Models</span>
    - saved models
- <span style="color:green"> Network.py</span>
- <span style="color:green"> Utils.py</span>
