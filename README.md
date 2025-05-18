
**My implementation of Proximal Policy Optimization (PPO) Reinforcement Learning algorithm from scratch, following the algorithm in the original OpenAI paper.**

To evaluate the performance of my custom PPO I benchmarked it with the stable baseline PPO. The results (on example of Acrobot-v1 environment
after 300000 of total training timesteps):

**Performance**

![](training_runs/20250517_042955/learning_curves.png)

Both agents learned to solve the game:

Stable baseline PPO:

<img src="sb3_ppo.gif" width="300" height="200">

My PPO:

<img src="my_ppo.gif" width="300" height="200">


Also there are very unfinished notes on RL (I will add new material I learned recently soon)

To be added soon (polishing the last details): 

- new topics to the notes
- parallelized version of ppo


**Related project:** 

gym style snake game: I created the customized env that is fully compatible with RL libraries and can be used interchangible with OpenAI's gym env

Available at https://github.com/TatianaUshakova/Gym_Style_Snake

to be added there
 - experiments with custom design of reward funcitions, structure of learning and partial information availability to identify ways of more efficient learning + notes-reflection about that
