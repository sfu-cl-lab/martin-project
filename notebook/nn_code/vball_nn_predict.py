from nn_lstm import vball_lstm
from data_utils import get_next_batch, prepare_data, shuffle_data
import tensorflow as tf
import numpy as np
import os
import time
import pickle
from sklearn import metrics
import pandas as pd

# config variables
FEATURE_COUNT = 41
LSTM_CELL_COUNT = 256
MAX_TRACE_LENGTH = 10
LEARNING_RATE = 1e-7 #1e-4
THRESHOLD = 0.0001 #0.0001
BATCH_SIZE = 32 #32
GAMMA = 1

SAVE_DIR = 'C:/projects/msc-project/output_retry_15'
DATA_FILE = 'C:/projects/msc-project/data/vb_data_3_numZone.csv'


def train_network(sess, model, state_data, reward_data, trace_lengths, reward_targets, episode_ids):

    # load original dataset (needed for markov error computation)
    vb = pd.read_csv(DATA_FILE)

    converge_flag = False

    merge = tf.summary.merge_all()
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    start_iteration = 1

    state_data_orig = state_data.copy()
    trace_lengths_orig = trace_lengths.copy()

    mse_errors = []

    restore_model = True
    if restore_model:
        checkpoint = tf.train.get_checkpoint_state(SAVE_DIR)
        if checkpoint and checkpoint.model_checkpoint_path:
            start_iteration = int((checkpoint.model_checkpoint_path.split("-"))[-1]) + 1
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights.")

    # initial Q values
    [rnn_outputs_prev, q_predictions_prev] = sess.run([model.outputs, model.read_out],
                                                    feed_dict={model.trace_lengths: trace_lengths_orig,
                                                            model.rnn_input: state_data_orig})

    pickle.dump(q_predictions_prev, open(SAVE_DIR + '/q_values_before.pkl', 'wb'))


    for iteration in range(start_iteration,251):

        t = time.time()

        print('Iteration %d ...' % (iteration))
        costs = []
        sample_index = 0

        state_data, reward_data, trace_lengths, episode_ids = shuffle_data(state_data, reward_data, trace_lengths, episode_ids)

        while sample_index < len(state_data):
            # prepare batch
            states0, states1, rewards, traces0, traces1, end_index = get_next_batch(state_data, reward_data, trace_lengths, sample_index, BATCH_SIZE)
            

            # compute NN prediction
            [rnn_outputs_t1, q_predictions_t1] = sess.run([model.outputs, model.read_out],
                                                    feed_dict={model.trace_lengths: traces1,
                                                            model.rnn_input: states1})
            q_target = np.zeros([len(rewards), 1])
            for i in range(len(rewards)):
                if rewards[i] != 0:
                    # in this case, we're at the end of an episode
                    q_target[i,0] = rewards[i]
                else:
                    q_target[i,0] = rewards[i] + GAMMA * q_predictions_t1[i,0]
            
            # update with gradient
            [diff, read_out, cost, summary_train, _] = sess.run(
                        [model.diff, model.read_out, model.cost, merge, model.train_step],
                        feed_dict={model.y: q_target,
                                model.trace_lengths: traces0,
                                model.rnn_input: states0})
            
            sample_index = end_index + 1
            costs.append(cost)

            #if sample_index >= 96794:
            #    print(read_out)

        # end of iteration loop
        
        iteration_cost = np.mean(costs)
        print('... done. Time elapsed: %.2f seconds' % (time.time() - t))
        print('Iteration cost: %.3f' % (iteration_cost))

        [rnn_outputs, q_predictions] = sess.run([model.outputs, model.read_out],
                                                    feed_dict={model.trace_lengths: trace_lengths_orig,
                                                            model.rnn_input: state_data_orig})
        

        # compute mean change in Q predictions
        avg_change = np.mean(np.abs(q_predictions - q_predictions_prev))
        avg_Q = np.mean(q_predictions)
        print('Mean change from last iteration: %.5f' % (avg_change))
        print('Mean Q value: %.5f' % (avg_Q))
        q_predictions_prev = q_predictions

        # clip and compute MSE error
        q_predictions[q_predictions > 1] = 1
        q_predictions[q_predictions < -1] = -1
        mse_error = metrics.mean_squared_error(reward_targets, q_predictions)
        print('MSE error: %.5f' % (mse_error))
        mse_errors.append(mse_error)

        # compute markov property error
        outcome_regex = '[\-\!\+]'
        outcome_cond0 = pd.Series(vb['ActionOutcome0']).str.match(outcome_regex)
        outcome_cond1 = pd.Series(vb['ActionOutcome1']).str.match(outcome_regex)
        cond0 = (vb['ActionType0']=='S') & (vb['ActionHome0']==1) & (outcome_cond0)
        cond1 = (vb['ActionType1']=='S') & (vb['ActionHome1']==1) & (outcome_cond1)
            
        markov_error = np.abs(np.mean(q_predictions[cond1]) - np.mean(q_predictions[cond0]))
        print('Markov error: %.5f' % (markov_error))

        saver.save(sess, SAVE_DIR + '/nn-iter', global_step=iteration)
        pickle.dump(mse_errors, open(SAVE_DIR + ('/mse_errors%.8f.pkl' % LEARNING_RATE), 'wb'))

        # check for convergence
        if avg_change < THRESHOLD:
            converge_flag = True
        
        if avg_change < 10*THRESHOLD:
            pickle.dump(q_predictions, open('%s/q_values_%03d.pkl' % (SAVE_DIR, iteration), 'wb'))

        if converge_flag:
            print('Convergence detected, exiting.')
            print('Saving Q values.')
            pickle.dump(q_predictions, open(SAVE_DIR + '/q_values.pkl', 'wb'))
            break

    print('Done.')


if __name__ == '__main__':
    if not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    sess = tf.InteractiveSession()
    nn = vball_lstm(FEATURE_COUNT, LSTM_CELL_COUNT, MAX_TRACE_LENGTH, LEARNING_RATE)
    
    print('Preparing data ...')
    state_data, reward_data, trace_lengths, reward_targets, episode_ids = prepare_data(DATA_FILE, max_trace_length=MAX_TRACE_LENGTH, save_dir=SAVE_DIR, load_from_saved=True)
    print('... done.')
    train_network(sess, nn, state_data, reward_data, trace_lengths, reward_targets, episode_ids)