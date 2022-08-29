import gym
import random
import numpy as np
import tflearn
from tflearn.layers.recurrent import lstm
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter
#import keras
#from matplotlib import pyplot as plt

# 1) as described in the methodology (parameters)
env = gym.make("CartPole-v1")
env.reset()
LR = 1e-3
goal_steps = 500
score_threshold = 50
episodes = 10000

# 2) Defining an episode
def random_games():
    for episode in range(5):
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break               
random_games()

# 3) Generating Training Data
def data_samples():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(episodes):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            action = random.randrange(0,2)
            observation, reward, done, info = env.step(action)
            if len(prev_observation) > 0 :
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score+=reward
            if done: break
        # analysing the game 
        if score >= score_threshold:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]
                    
                # saving our training data
                training_data.append([data[0], output])
        # reset env to play again
        env.reset()
        # save overall scores
        scores.append(score)
    
    training_data_save = np.array(training_data)
    np.save('saved.npy',training_data_save)
    print('Average accepted score:',mean(accepted_scores))
    print('Median score for accepted scores:',median(accepted_scores))
    print(Counter(accepted_scores))
    return training_data

# 4) NN model desgin (RNN)
def NN(input_size):

    net = input_data(shape=(None, input_size, 1), name='input')
    net = dropout(net, 0.8)
    net = lstm(net, 128, activation='relu', return_seq=True)
    net = dropout(net, 0.8)
    net = lstm(net, 128, activation='relu', return_seq=True)
    net = dropout(net, 0.8)
    net = fully_connected(net, 2, activation='linear')
    net = regression(net, optimizer='adam', learning_rate=LR, loss='mean_square', name='targets')
    model = tflearn.DNN(net, tensorboard_verbose=2)
     
    return model

# 5) Training Model
def train_model(training_data, model=False):
    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    y = [i[1] for i in training_data]
    if not model:
        model = NN(input_size = len(X[0]))
    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openai_learning')
    return model

# produce training data:
training_data = data_samples()
model = train_model(training_data)

# 6) Testing Model
scores = []
choices = []
for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(goal_steps):
        env.render()
        if len(prev_obs)==0:
            action = random.randrange(0,2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
        choices.append(action)
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score+=reward
        if done: break
    scores.append(score)

print('Average Score:',sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
print(score_threshold)

