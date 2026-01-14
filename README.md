# Delayed Gratification Q-Learning Simulation

**Author:** Kary Zheng  
**Date:** January 14, 2026

This project implements a Q-learning agent that models delayed gratification behavior inspired by animal decision-making experiments. The agent learns to choose between immediate low-value rewards and delayed high-value rewards, with performance evaluated across increasing delay durations.

---

## Overview

The simulation tests whether a reinforcement learning agent will wait for a better reward as delays increase, and whether this behavior differs from control conditions where waiting provides no benefit.

The model separates:
- **Learning** of reward contingencies
- **Testing** under delay-based patience constraints

---

## Experimental Design

### States

The environment contains four possible states:

| State | Description |
|------|------------|
| EXPM_LR | Live shrimp on LEFT, dead shrimp on RIGHT |
| EXPM_RL | Live shrimp on RIGHT, dead shrimp on LEFT |
| CTRL_LR | Unobtainable shrimp on LEFT, dead shrimp on RIGHT |
| CTRL_RL | Unobtainable shrimp on RIGHT, dead shrimp on LEFT |

- **Experimental states** test delayed gratification.
- **Control states** ensure baseline discrimination between dead and unobtainable rewards.

---

### Rewards

| Outcome | Reward |
|------|------|
| Live shrimp | 5.0 |
| Dead shrimp | 1.0 |
| Unobtainable | 0.0 |

---

### Learning Algorithm

- Q-learning with learning rate `α = 0.10`
- Discount factor `γ = 0.99`
- Softmax action selection (`β = 1.0`)
- Q-table size: 4 states × 2 actions (LEFT, RIGHT)

---

### Delay and Patience Model

Delay values range from **10 to 130 seconds**.

Delay is converted into a probability of waiting using a cumulative normal distribution:

- Short delays → high probability of waiting
- Long delays → low probability of waiting

This probability discounts the Q-value of **live shrimp choices only**, modeling patience without affecting baseline preferences.

---

## Procedure

1. **Training Phase**
   - The agent learns reward contingencies across all states.
   - Performance is tracked separately for each state.

2. **Testing Phase**
   - For each delay value:
     - 100 trials are run
     - Experimental and control trials are analyzed separately
   - Percent correct choices are computed for each condition.

---

## Output

The simulation produces:
- Learning curves by state
- Percent correct vs delay plots
- Final Q-table values

These outputs demonstrate decreasing willingness to wait as delay increases in experimental trials, while control performance remains stable.

---

## Key Contribution

This implementation cleanly isolates delayed gratification effects by:
- Applying patience only to delayed rewards
- Separating experimental and control conditions
- Avoiding delay contamination of baseline learning

This makes the results interpretable and suitable for comparison with behavioral experiments.
