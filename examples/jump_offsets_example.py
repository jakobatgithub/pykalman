"""
Example: Using Jump Offsets (Dirac Delta Functions) with pykalman
===================================================================

This example demonstrates how to use jump offsets to model instantaneous
state changes in a Kalman filter. Jump offsets are useful for modeling:
- Sudden disturbances or perturbations
- Feeding events in bioreactors
- Dilution events
- Step changes in system state
- Any discrete events affecting continuous systems
"""

import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter


def example_constant_jump():
    """
    Example 1: Constant jump offset applied at each timestep.
    
    Models a system where a constant disturbance is applied at each time step,
    such as a bioreactor with continuous feeding.
    """
    print("Example 1: Constant Jump Offset")
    print("-" * 50)
    
    # System parameters - simple 1D system
    transition_matrix = np.array([[1.0]])
    observation_matrix = np.array([[1.0]])
    transition_covariance = np.array([[0.1]])
    observation_covariance = np.array([[0.1]])
    initial_state_mean = np.array([0.0])
    initial_state_covariance = np.array([[1.0]])
    
    # Constant jump of 0.5 at each transition
    jump_offsets = np.array([0.5])
    
    # Generate synthetic observations
    n_timesteps = 20
    np.random.seed(42)
    true_state = 0.0
    observations = []
    for t in range(n_timesteps):
        # State evolves with jump and noise
        true_state = true_state + 0.5 + np.random.randn() * 0.316
        obs = true_state + np.random.randn() * 0.316
        observations.append([obs])
    observations = np.array(observations)
    
    # Filter with jumps
    kf_with_jumps = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance,
        initial_state_mean=initial_state_mean,
        initial_state_covariance=initial_state_covariance,
        jump_offsets=jump_offsets,
    )
    filtered_with_jumps, _ = kf_with_jumps.filter(observations)
    
    # Filter without jumps for comparison
    kf_no_jumps = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance,
        initial_state_mean=initial_state_mean,
        initial_state_covariance=initial_state_covariance,
    )
    filtered_no_jumps, _ = kf_no_jumps.filter(observations)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(observations, 'ko', label='Observations', alpha=0.5)
    plt.plot(filtered_with_jumps, 'b-', label='Filtered (with jumps)', linewidth=2)
    plt.plot(filtered_no_jumps, 'r--', label='Filtered (no jumps)', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('State')
    plt.title('Constant Jump Offset Example')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/tmp/constant_jump_example.png', dpi=150)
    print("Plot saved to /tmp/constant_jump_example.png")
    print()


def example_time_varying_jump():
    """
    Example 2: Time-varying jump offsets.
    
    Models a system with discrete feeding events at specific times,
    such as intermittent feeding in a fed-batch bioreactor.
    """
    print("Example 2: Time-Varying Jump Offsets")
    print("-" * 50)
    
    # System parameters
    transition_matrix = np.array([[1.0]])
    observation_matrix = np.array([[1.0]])
    transition_covariance = np.array([[0.01]])
    observation_covariance = np.array([[0.1]])
    initial_state_mean = np.array([0.0])
    initial_state_covariance = np.array([[1.0]])
    
    # Time-varying jumps: feeding events at t=5, 10, 15
    n_timesteps = 20
    jump_offsets = np.zeros((n_timesteps - 1, 1))
    jump_offsets[4] = 2.0   # Large jump at t=5
    jump_offsets[9] = 1.5   # Medium jump at t=10
    jump_offsets[14] = 1.0  # Small jump at t=15
    
    # Generate observations with feeding events
    np.random.seed(42)
    true_state = 0.0
    observations = []
    for t in range(n_timesteps):
        if t > 0:
            true_state = true_state + jump_offsets[t-1, 0] + np.random.randn() * 0.1
        obs = true_state + np.random.randn() * 0.316
        observations.append([obs])
    observations = np.array(observations)
    
    # Filter with jumps
    kf = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance,
        initial_state_mean=initial_state_mean,
        initial_state_covariance=initial_state_covariance,
        jump_offsets=jump_offsets,
    )
    filtered_states, filtered_covariances = kf.filter(observations)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(observations, 'ko', label='Observations', alpha=0.5)
    plt.plot(filtered_states, 'b-', label='Filtered State', linewidth=2)
    
    # Mark feeding events
    feeding_times = [5, 10, 15]
    for ft in feeding_times:
        plt.axvline(ft, color='r', linestyle='--', alpha=0.5, linewidth=1)
    plt.text(5, plt.ylim()[1] * 0.9, 'Feed', ha='center', color='r')
    plt.text(10, plt.ylim()[1] * 0.9, 'Feed', ha='center', color='r')
    plt.text(15, plt.ylim()[1] * 0.9, 'Feed', ha='center', color='r')
    
    plt.xlabel('Time Step')
    plt.ylabel('State')
    plt.title('Time-Varying Jump Offsets Example (Feeding Events)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/tmp/time_varying_jump_example.png', dpi=150)
    print("Plot saved to /tmp/time_varying_jump_example.png")
    print()


def example_multidimensional_jump():
    """
    Example 3: Multi-dimensional state with different jumps per dimension.
    
    Models a 2D system where each state component has different jump dynamics.
    """
    print("Example 3: Multi-Dimensional Jump Offsets")
    print("-" * 50)
    
    # 2D system
    n_dim_state = 2
    n_dim_obs = 2
    transition_matrix = np.eye(n_dim_state)
    observation_matrix = np.eye(n_dim_obs)
    transition_covariance = 0.05 * np.eye(n_dim_state)
    observation_covariance = 0.1 * np.eye(n_dim_obs)
    initial_state_mean = np.zeros(n_dim_state)
    initial_state_covariance = np.eye(n_dim_state)
    
    # Different jumps for each dimension
    # Dimension 0: positive jump (increase)
    # Dimension 1: negative jump (decrease)
    jump_offsets = np.array([0.3, -0.2])
    
    # Generate observations
    n_timesteps = 30
    np.random.seed(42)
    true_state = np.zeros(n_dim_state)
    observations = []
    for t in range(n_timesteps):
        if t > 0:
            true_state = true_state + jump_offsets + np.random.randn(n_dim_state) * 0.22
        obs = true_state + np.random.randn(n_dim_obs) * 0.316
        observations.append(obs)
    observations = np.array(observations)
    
    # Filter
    kf = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance,
        initial_state_mean=initial_state_mean,
        initial_state_covariance=initial_state_covariance,
        jump_offsets=jump_offsets,
    )
    filtered_states, _ = kf.filter(observations)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Dimension 0 (increasing)
    ax1.plot(observations[:, 0], 'ko', label='Observations', alpha=0.5)
    ax1.plot(filtered_states[:, 0], 'b-', label='Filtered State', linewidth=2)
    ax1.set_ylabel('State Dimension 0')
    ax1.set_title('Dimension 0: Positive Jump (+0.3)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Dimension 1 (decreasing)
    ax2.plot(observations[:, 1], 'ko', label='Observations', alpha=0.5)
    ax2.plot(filtered_states[:, 1], 'r-', label='Filtered State', linewidth=2)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('State Dimension 1')
    ax2.set_title('Dimension 1: Negative Jump (-0.2)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/multidim_jump_example.png', dpi=150)
    print("Plot saved to /tmp/multidim_jump_example.png")
    print()


if __name__ == '__main__':
    print("=" * 60)
    print("Jump Offsets (Dirac Delta Functions) Examples")
    print("=" * 60)
    print()
    
    example_constant_jump()
    example_time_varying_jump()
    example_multidimensional_jump()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
