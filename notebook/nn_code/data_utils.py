import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import MinMaxScaler

def prepare_data(filename, max_trace_length=10, save_dir='', load_from_saved=True):

    if (save_dir != '') & (load_from_saved):
        if os.path.isfile(save_dir + '/state_data.pkl'):
            state_data = pickle.load(open(save_dir + '/state_data.pkl', 'rb'))
            reward_data = pickle.load(open(save_dir + '/reward_data.pkl', 'rb'))
            trace_lengths = pickle.load(open(save_dir + '/trace_lengths.pkl', 'rb'))
            return state_data, reward_data, trace_lengths
        else:
            print('Error loading saved data, reverting to original dataset.')

    # load data
    vb = pd.read_csv(filename)

    # drop unused columns and get dummy columns for categorical variables
    X = vb.drop(['Season', 'GameID', 'PlayerTeam', 'PlayerName', 'RewardDistance', 'RewardValue'], axis=1)
    cols = [col for col in list(X.columns) if X[col].dtype == 'object']
    X = pd.get_dummies(data=X, columns=cols)

    # drop action history columns (only last action kept)
    cols = [col for col in X.columns if ('Action' in col) & ('0' in col)]
    cols.append('SetNumber')
    cols.append('ScoreMax')
    cols.append('ScoreDiff')
    XX = X[cols]

    # scale data
    scaler = MinMaxScaler()
    scaler.fit(XX)
    XX = scaler.transform(XX)

    # format data for neural net
    state_data = np.zeros([len(XX), 10, 41])
    trace_lengths = np.zeros([len(XX)])
    reward_data = np.zeros([len(XX)])

    prev_reward_dist = -1
    lookback = 0
    max_lookback = max_trace_length-1

    for i in range(len(XX)):
        vb_row = vb.iloc[i]
        XX_rows = XX[max(0,i-9):i+1,:]
        
        if vb_row.RewardDistance >= prev_reward_dist: #reward distance not decreasing means new rally
            lookback = 0
        else:
            lookback = min(lookback+1, max_lookback)
            
        lookback_frame = XX_rows[len(XX_rows)-lookback-1:,:]
        state_data[i,0:len(lookback_frame),:] = lookback_frame
        
        trace_lengths[i] = lookback+1
        
        if vb_row.RewardDistance == 0:
            reward_data[i] = vb_row.RewardValue
        
        prev_reward_dist = vb_row.RewardDistance

        if (i+1) % 10000 == 0:
            print('%d/%d' % (i+1, len(XX)))

    if os.path.isdir(save_dir):
        pickle.dump(state_data, open(save_dir + '/state_data.pkl', 'wb'))
        pickle.dump(reward_data, open(save_dir + '/reward_data.pkl', 'wb'))
        pickle.dump(trace_lengths, open(save_dir + '/trace_lengths.pkl', 'wb'))

    return state_data, reward_data, trace_lengths



def get_next_batch(state_data, reward_data, state_trace_lengths, start_idx, BATCH_SIZE):

    last_idx = len(state_data) - 1

    if start_idx + BATCH_SIZE < len(state_data):
        end_idx = start_idx + BATCH_SIZE - 1
    else:
        end_idx = last_idx

    while reward_data[end_idx] == 0:
        end_idx -= 1

    states0 = state_data[start_idx:end_idx+1,:,:]
    rewards = reward_data[start_idx:end_idx+1]
    traces0 = state_trace_lengths[start_idx:end_idx+1]

    if end_idx != last_idx:
        states1 = state_data[start_idx+1:end_idx+2,:,:]
        traces1 = state_trace_lengths[start_idx+1:end_idx+2]
    else:
        states1 = state_data[start_idx+1:last_idx+1,:,:]
        states1 = np.append(states1, state_data[last_idx:last_idx+1,:,:], axis=0)
        traces1 = state_trace_lengths[start_idx+1:last_idx+1]
        traces1 = np.append(traces1, state_trace_lengths[last_idx])

    return states0, states1, rewards, traces0, traces1, end_idx