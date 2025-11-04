"""
======================================================
Using Control Inputs with the Unscented Kalman Filter
======================================================

This example demonstrates how to use control inputs (also known as external 
inputs or known actions) with the Unscented Kalman Filter (UKF).

Control inputs are useful when modeling systems where you know about external
forces or actions that affect the system state. For example:
- Acceleration commands in a vehicle tracking system
- Voltage inputs to a motor
- Control signals to a robotic arm

The system modeled here is a simple 1D position-velocity system with control:
    x_{t+1} = A*x_t + B*u_t + noise
    z_t = C*x_t + noise

where:
    x = [position, velocity]
    u = [acceleration command]
    z = [position measurement]
"""

import matplotlib.pyplot as plt
import numpy as np

from pykalman import UnscentedKalmanFilter

# Define system matrices for a position-velocity model
A = np.array([[1, 1],    # position += velocity
              [0, 1]])    # velocity unchanged (without input)

B = np.array([[0.5],     # position affected by 0.5 * acceleration
              [1]])       # velocity affected by acceleration

C = np.array([[1, 0]])   # observe position only

# Define transition and observation functions
def transition_function(state, control_input, noise):
    """State transition with control input and process noise."""
    return A.dot(state) + B.dot(control_input) + noise

def observation_function(state, noise):
    """Observation function with observation noise."""
    return C.dot(state) + noise

# Set up the UKF
n_timesteps = 50
initial_state = np.array([0, 0])  # start at position=0, velocity=0
initial_covariance = np.eye(2)
process_noise = np.eye(2) * 0.01
observation_noise = np.array([[0.5]])

# Create control input sequence (a sinusoidal acceleration pattern)
time = np.arange(n_timesteps - 1)
control_inputs = 0.5 * np.sin(time * 0.2).reshape(-1, 1)

# Create the UKF with control inputs
ukf = UnscentedKalmanFilter(
    transition_functions=transition_function,
    observation_functions=observation_function,
    transition_covariance=process_noise,
    observation_covariance=observation_noise,
    initial_state_mean=initial_state,
    initial_state_covariance=initial_covariance,
    transition_inputs=control_inputs,
    random_state=42
)

# Generate true states and noisy observations
true_states, observations = ukf.sample(
    n_timesteps,
    initial_state=initial_state,
    U=control_inputs
)

# Run the UKF with control inputs
filtered_states, filtered_covariances = ukf.filter(observations, U=control_inputs)

# Run the UKF smoother with control inputs
smoothed_states, smoothed_covariances = ukf.smooth(observations, U=control_inputs)

# For comparison, run UKF without control inputs to show the difference
ukf_no_input = UnscentedKalmanFilter(
    transition_functions=lambda s, n: A.dot(s) + n,  # no control input
    observation_functions=observation_function,
    transition_covariance=process_noise,
    observation_covariance=observation_noise,
    initial_state_mean=initial_state,
    initial_state_covariance=initial_covariance,
)
filtered_no_input, _ = ukf_no_input.filter(observations)

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

# Plot 1: Position
ax1.plot(true_states[:, 0], 'b-', label='True Position', linewidth=2)
ax1.plot(observations[:, 0], 'k.', label='Observations', markersize=4, alpha=0.5)
ax1.plot(filtered_states[:, 0], 'r--', label='Filtered (with input)', linewidth=2)
ax1.plot(smoothed_states[:, 0], 'g-.', label='Smoothed (with input)', linewidth=2)
ax1.plot(filtered_no_input[:, 0], 'm:', label='Filtered (no input)', linewidth=2)
ax1.set_ylabel('Position')
ax1.set_title('UKF with Control Inputs - Position Tracking')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# Plot 2: Velocity
ax2.plot(true_states[:, 1], 'b-', label='True Velocity', linewidth=2)
ax2.plot(filtered_states[:, 1], 'r--', label='Filtered (with input)', linewidth=2)
ax2.plot(smoothed_states[:, 1], 'g-.', label='Smoothed (with input)', linewidth=2)
ax2.plot(filtered_no_input[:, 1], 'm:', label='Filtered (no input)', linewidth=2)
ax2.set_ylabel('Velocity')
ax2.set_title('Velocity Estimation')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

# Plot 3: Control inputs
ax3.plot(control_inputs, 'c-', label='Control Input (Acceleration)', linewidth=2)
ax3.set_xlabel('Time Step')
ax3.set_ylabel('Acceleration')
ax3.set_title('Control Input Signal')
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ukf_control_inputs.png', dpi=150, bbox_inches='tight')
print("Plot saved as 'ukf_control_inputs.png'")

# Calculate and print errors
position_error_with_input = np.abs(filtered_states[:, 0] - true_states[:, 0])
position_error_no_input = np.abs(filtered_no_input[:, 0] - true_states[:, 0])

print("\nPerformance Comparison:")
print(f"Mean Position Error (with control input): {np.mean(position_error_with_input):.4f}")
print(f"Mean Position Error (without control input): {np.mean(position_error_no_input):.4f}")
print(f"Improvement: {(1 - np.mean(position_error_with_input)/np.mean(position_error_no_input))*100:.1f}%")

plt.show()
