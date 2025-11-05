"""
====================================================
Applying the Unscented Kalman Filter with Missing Observations
====================================================

This example shows how one may apply :class:`UnscentedKalmanFilter` and 
:class:`AdditiveUnscentedKalmanFilter` when some measurements are missing.

While Kalman Filters are typically presented assuming a measurement exists 
for every time step, this is not always the case in reality. The Unscented
Kalman Filters are now implemented to recognize masked portions of numpy
arrays as missing measurements.

The figure drawn illustrates the trajectory of each dimension of the true
state, the estimated state using all measurements, and the estimated state
using every fifth measurement for both the general and additive versions
of the Unscented Kalman Filter.
"""

import numpy as np
import matplotlib.pyplot as plt
from pykalman import AdditiveUnscentedKalmanFilter

# Specify parameters
random_state = np.random.RandomState(0)
n_timesteps = 50

# Define transition and observation functions
transition_matrix = np.array([[1, 0.1], [0, 1]])
observation_matrix = np.eye(2) + random_state.randn(2, 2) * 0.1

# Additive noise
transition_covariance = np.eye(2) * 0.01
observation_covariance = np.eye(2) * 0.1

# Initial state
initial_state_mean = [5, -5]
initial_state_covariance = np.eye(2)


def transition_function(state):
    """Transition function for additive UKF."""
    return transition_matrix.dot(state)


def observation_function(state):
    """Observation function for additive UKF."""
    return observation_matrix.dot(state)


# Create the filter
kf = AdditiveUnscentedKalmanFilter(
    transition_functions=transition_function,
    observation_functions=observation_function,
    transition_covariance=transition_covariance,
    observation_covariance=observation_covariance,
    initial_state_mean=initial_state_mean,
    initial_state_covariance=initial_state_covariance,
    random_state=0
)

# Sample from the model
states, observations_all = kf.sample(
    n_timesteps, initial_state=initial_state_mean
)

# Label 4/5 of the observations as missing
observations_missing = np.ma.array(
    observations_all,
    mask=np.zeros(observations_all.shape)
)
for t in range(n_timesteps):
    if t % 5 != 0:
        observations_missing[t] = np.ma.masked

# Estimate state with filtering using all observations
filtered_states_all = kf.filter(observations_all)[0]

# Estimate state with filtering using sparse observations
filtered_states_missing = kf.filter(observations_missing)[0]

# Draw estimates
plt.figure(figsize=(12, 6))

# Plot first dimension
plt.subplot(1, 2, 1)
plt.plot(states[:, 0], 'b-', label='True State', linewidth=2)
plt.plot(filtered_states_all[:, 0], 'r--', label='All Observations', linewidth=2)
plt.plot(filtered_states_missing[:, 0], 'g:', label='Sparse Observations', linewidth=2)
plt.xlabel('Time Step')
plt.ylabel('State Dimension 1')
plt.title('Unscented Kalman Filter with Missing Observations')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

# Plot second dimension
plt.subplot(1, 2, 2)
plt.plot(states[:, 1], 'b-', label='True State', linewidth=2)
plt.plot(filtered_states_all[:, 1], 'r--', label='All Observations', linewidth=2)
plt.plot(filtered_states_missing[:, 1], 'g:', label='Sparse Observations', linewidth=2)
plt.xlabel('Time Step')
plt.ylabel('State Dimension 2')
plt.title('Unscented Kalman Filter with Missing Observations')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ukf_missing_observations.png', dpi=150, bbox_inches='tight')
print("Plot saved to ukf_missing_observations.png")
plt.show()
