import numpy as np
import gym

# Setting environment
environment = gym.make('CartPole-v0')

# Setting initial variables
bestWeights = np.zeros(4)
bestReward = 0
lenght = []


for _ in range(2000):
    newReward = 0
    observation = environment.reset()
    done = False

    # Randomly setting new weights
    newWeights = np.random.uniform(-1,1,4)
    while not done:
        action = 1 if np.dot(observation,newWeights) > 0 else 0
        observation, reward,done, _= environment.step(action)
        newReward += reward

    # Updating Weights
    if newReward > bestReward:
        bestReward = newReward
        bestWeights = newWeights


# Create screen and show some episodes running after the learning process
for _ in range(2000):

    observation = environment.reset()
    done = False
    while not done:
        environment.render()
        action = 1 if np.dot(observation,bestWeights) > 0 else 0
        observation, reward,done,_= environment.step(action)
    