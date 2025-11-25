import numpy as np


def sample_opponent_policy_two(strategies=('allc', 'alld')):
    probs = np.array([0.50, 0.50])
    return np.random.choice(list(strategies), p=probs)


def sample_opponent_policy_three(strategies=('allc', 'alld', 'tft')):
    probs = np.array([0.33, 0.34, 0.33])
    return np.random.choice(list(strategies), p=probs)


def CL_sample_opponent_policy_three(
    time_step,
    max_training_timesteps=2e7,
    strategies=('allc', 'alld', 'tft'),
    phase_A=0.20,
    phase_B=0.50
):
    frac = min(1.0, time_step / max_training_timesteps)
    if frac <= phase_A:
        probs = np.array([0.80, 0.10, 0.10])  # Early: mostly AllC
    elif frac <= phase_B:
        probs = np.array([0.35, 0.35, 0.30])  # Mid: mix in AllD/TFT
    else:
        probs = np.array([0.20, 0.40, 0.40])  # Late: more AllD
    return np.random.choice(list(strategies), p=probs)
