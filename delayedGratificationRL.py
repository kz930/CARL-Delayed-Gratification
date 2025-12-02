import random
from math import *
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


'''
Jeff Krichmar
30 November 2025

Delayed gratification using Q learning and patience 

'''

ACTIONS = 2
LEFT = 0
RIGHT = 1

STATES = 4  
EXPM_LR = 0  # Live Shrimp (left); Dead Shrimp (right)
EXPM_RL = 1  # Live Shrimp (right); Dead Shrimp (left)
CTRL_LR = 2  # Unobtainable Shrimp (left); Dead Shrimp (right)
CTRL_RL = 3  # Unobtainable Shrimp (right); Dead Shrimp (left)

LIVE_RWD = 5.0
DEAD_RWD = 1.0
UNOBTAINABLE_RWD = 0.0

# Learning parameters
ALPHA = 0.10
BETA = 1.0
GAMMA = 0.99

# initialize the Q table in a state by action matrix
qTbl = [[0.0 for y in range(ACTIONS)] for x in range(STATES)]


def prob_wait(tim):
    '''
    Calculate the probability to wait. Uses a cumulative normal distribution and exponential.
    Slightly different from the equation in Xing et al.
    :param tim: elapsed time in seconds
    :return: probability based on the time elapsed, where 0 would mean do not wait and 1 equates to 100% chance of
             waiting, and 0.5 means that there is a 50% chance of waiting.
    '''
    mean = 70
    std_dev = 20
    beta_weight = 2
    pw = 1 / np.exp(beta_weight*norm.cdf(tim, loc=mean, scale=std_dev))
    return pw


def action_select(q, beta):
    """
       Calculate the Softmax function to choose an action. Converts the q array into a probability distribution
       - Parameters
           q - expected values
           beta - temperature for Softmax function
       - Returns the selected action
    """
    act = 0
    p = 0
    sumSoftMax = 0
    sumP = 0

    # calculate the denominator sum
    for i in range(len(q)):
        sumSoftMax += exp(beta*q[i])

    r = random.random() # get a random number between 0 and 1
    done = False

    # loop through the q values
    for i in range(len(q)):
        # add the softmax probability to the sum total.
        # if the sum is greater than the total, that action is chosen.
        if not done:
            p = exp(beta*q[i])/sumSoftMax
            sumP += p
            if sumP >= r:
                done = True
                act = i
    return act


def print_qtbl(Q):
    """
       Prints out the Q table and the corresponding probabilities
       - Parameters
           Q - expected values for each state
           t - the trial number
    """
    print("")
    print('State Q(LEFT) Q(RIGHT)')
    for i in range(len(Q)):
        print("%d      %3.2f     %3.2f" % (i, Q[i][0], Q[i][1]))


def run_trial(p_wait):
    '''
    Runs a single delayed gratification trial.
    :param p_wait: Probability of waiting for the live shrimp to be available.  p_wait is multiplied with the Q value
                   associated with choosing a live shrimp.  A p_wait value of 1.0 means that the choice is not
                   dependent on the time elapsed.
    :return: Reward value based on the action selection, and a boolean that is true if this was an experimental trial
    '''

    q = qTbl  # local copy of Q table

    # Randomly choose an experimental or control trial. Randomize the location of the live shrimp.
    # Get the Q value. Decrease the Q value by the probability to wait
    if random.random() < 0.25:
        current_state = EXPM_LR
        q[current_state][LEFT] = q[current_state][LEFT] * p_wait
    elif random.random() < 0.5:
        current_state = EXPM_RL
        q[current_state][RIGHT] = q[current_state][RIGHT] * p_wait
    elif random.random() >= 0.75:
        current_state = CTRL_LR
        q[current_state][LEFT] = q[current_state][LEFT] * p_wait
    else:
        current_state = CTRL_RL
        q[current_state][RIGHT] = q[current_state][RIGHT] * p_wait

    act = action_select(q[current_state], BETA)  # select an action based on the state and Q value

    # Based on the state and action, return the appropriate reward
    if current_state == EXPM_LR:
        if act == LEFT:
            rwd = LIVE_RWD
        else:
            rwd = DEAD_RWD
    elif current_state == EXPM_RL:
        if act == LEFT:
            rwd = DEAD_RWD
        else:
            rwd = LIVE_RWD
    elif current_state == CTRL_LR:
        if act == LEFT:
            rwd = UNOBTAINABLE_RWD
        else:
            rwd = DEAD_RWD
    else:
        if act == LEFT:
            rwd = DEAD_RWD
        else:
            rwd = UNOBTAINABLE_RWD

    # print("%d\t%d\t%d\t%3.2f" % (t, current_state, act, rwd))

    # Update state-action table Q.
    qTbl[current_state][act] = qTbl[current_state][act] + ALPHA*(rwd - qTbl[current_state][act])

    if current_state < CTRL_LR:
        experimental_trial = True
    else:
        experimental_trial = False

    return rwd, experimental_trial

"""
    MAIN ROUTINE
"""

# First, run enough trials so the agent learns live, dead, and unobtainable shrimp preferences
TRIALS = 100
for t in range(TRIALS):
    run_trial(1.0)
# print_qtbl(qTbl)

# Trials with delays ranging from 10 seconds to 130 seconds
# For now, run 100 trials so there is enough of each trial type.
DELAY_TRIALS = 100
delays = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130])

# initialize results arrays for the experimental and control trials
delay_gratification_experiment_results = np.zeros([delays.shape[0]])
delay_gratification_ctrl_results = np.zeros([delays.shape[0]])

print("delay\texpm\tctrl")

# For each delay, run 100 trials. The run_trial takes the probablity to wait based on the delay as input.
for d in range(delays.shape[0]):
    exp_cnt = 0
    ctrl_cnt = 0
    for t in range(DELAY_TRIALS):
        r, ex = run_trial(prob_wait(delays[d]))
        if ex:
            exp_cnt += 1
            delay_gratification_experiment_results[d] += int(r > DEAD_RWD)
        else:
            ctrl_cnt += 1
            delay_gratification_ctrl_results[d] += int(r > UNOBTAINABLE_RWD)

    print("%d\t%3.2f\t%3.2f" % (delays[d], (delay_gratification_experiment_results[d]*100)/exp_cnt, delay_gratification_ctrl_results[d]*100/ctrl_cnt))

# plt.boxplot(delay_gratification_experiment_results, widths=0.75, tick_labels=['10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130'])
# plt.show()

# ----- PLOT RESULTS -----
plt.figure(figsize=(10, 6))
exp_percent = 100 * delay_gratification_experiment_results / DELAY_TRIALS
ctrl_percent = 100 * delay_gratification_ctrl_results / DELAY_TRIALS
plt.plot(delays, exp_percent, marker='o', linewidth=2, label="Experimental")
plt.plot(delays, ctrl_percent, marker='s', linewidth=2, label="Control")
plt.xlabel("Delay (seconds)")
plt.ylabel("Percent Choosing Better Option (%)")
plt.title("Delayed Gratification Performance vs. Delay")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()
