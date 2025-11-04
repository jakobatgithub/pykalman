"""
Example demonstrating jump offset support in unscented Kalman filters.

This example shows how to use jump offsets (Dirac Delta-like inputs) with
the UnscentedKalmanFilter to track a system that experiences sudden jumps
in its state.
"""

import numpy as np
from pykalman import UnscentedKalmanFilter

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def example_1_constant_jump():
    """Example 1: Constant jump offset at each timestep."""
    print("Example 1: Constant jump offset")
    print("-" * 50)
    
    n_timesteps = 20
    
    # System with constant upward jump of 0.3 at each transition
    jump_offsets = np.array([0.3])
    
    # Define simple transition and observation functions
    def transition_function(state, noise):
        return state + noise
    
    def observation_function(state, noise):
        return state + noise
    
    # Create filter with jump offsets
    ukf = UnscentedKalmanFilter(
        transition_functions=transition_function,
        observation_functions=observation_function,
        transition_covariance=np.array([[0.1]]),
        observation_covariance=np.array([[0.1]]),
        initial_state_mean=np.array([0.0]),
        initial_state_covariance=np.array([[1.0]]),
        jump_offsets=jump_offsets,
        random_state=42,
    )
    
    # Sample from the system
    states, observations = ukf.sample(n_timesteps)
    
    # Filter the observations
    filtered_means, filtered_covs = ukf.filter(observations)
    
    print(f"Initial state: {states[0, 0]:.3f}")
    print(f"Final state: {states[-1, 0]:.3f}")
    print(f"Expected cumulative jump: {jump_offsets[0] * (n_timesteps - 1):.3f}")
    print(f"Actual state increase: {states[-1, 0] - states[0, 0]:.3f}")
    print()


def example_2_time_varying_jumps():
    """Example 2: Time-varying jump offsets with occasional large jumps."""
    print("Example 2: Time-varying jump offsets")
    print("-" * 50)
    
    n_timesteps = 20
    
    # Create jump offsets with occasional large jumps
    jump_offsets = np.zeros((n_timesteps - 1, 1))
    jump_offsets[4] = 2.0   # Large jump at t=5
    jump_offsets[9] = 1.5   # Medium jump at t=10
    jump_offsets[14] = 1.0  # Small jump at t=15
    
    # Define transition and observation functions
    def transition_function(state, noise):
        return state + noise
    
    def observation_function(state, noise):
        return state + noise
    
    # Create filter with time-varying jumps
    ukf = UnscentedKalmanFilter(
        transition_functions=transition_function,
        observation_functions=observation_function,
        transition_covariance=np.array([[0.1]]),
        observation_covariance=np.array([[0.1]]),
        initial_state_mean=np.array([0.0]),
        initial_state_covariance=np.array([[1.0]]),
        jump_offsets=jump_offsets,
        random_state=42,
    )
    
    # Sample from the system
    states, observations = ukf.sample(n_timesteps)
    
    # Filter with and without jump offsets
    filtered_with_jumps, _ = ukf.filter(observations)
    
    # Create filter without jumps for comparison
    ukf_no_jumps = UnscentedKalmanFilter(
        transition_functions=transition_function,
        observation_functions=observation_function,
        transition_covariance=np.array([[0.1]]),
        observation_covariance=np.array([[0.1]]),
        initial_state_mean=np.array([0.0]),
        initial_state_covariance=np.array([[1.0]]),
    )
    filtered_no_jumps, _ = ukf_no_jumps.filter(observations)
    
    print(f"True final state: {states[-1, 0]:.3f}")
    print(f"Filtered state (with jumps): {filtered_with_jumps[-1, 0]:.3f}")
    print(f"Filtered state (no jumps): {filtered_no_jumps[-1, 0]:.3f}")
    print(f"Total jump magnitude: {np.sum(jump_offsets):.3f}")
    print()
    
    # Visualize if matplotlib is available
    if MATPLOTLIB_AVAILABLE:
        try:
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.plot(states, 'g-', label='True state', linewidth=2)
            plt.plot(observations, 'k.', alpha=0.3, label='Observations')
            plt.plot(filtered_with_jumps, 'b-', label='Filtered (with jumps)', linewidth=2)
            plt.plot(filtered_no_jumps, 'r--', label='Filtered (no jumps)', linewidth=2)
            
            # Mark the jump locations
            for i, jump in enumerate(jump_offsets):
                if jump[0] > 0:
                    plt.axvline(x=i+1, color='orange', linestyle=':', alpha=0.5)
                    plt.text(i+1, plt.ylim()[0], f'Jump\n{jump[0]:.1f}', 
                            ha='center', va='bottom', fontsize=8)
            
            plt.xlabel('Time step')
            plt.ylabel('State')
            plt.title('Unscented Kalman Filter with Jump Offsets')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.bar(range(len(jump_offsets)), jump_offsets.flatten())
            plt.xlabel('Time step')
            plt.ylabel('Jump magnitude')
            plt.title('Jump Offsets Over Time')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('/tmp/unscented_jump_offsets_example.png', dpi=100)
            print("Plot saved to /tmp/unscented_jump_offsets_example.png")
        except Exception as e:
            print(f"Could not create plot: {e}")
    else:
        print("Matplotlib not available, skipping plot generation")


def example_3_multidimensional():
    """Example 3: Multi-dimensional system with different jumps per dimension."""
    print("Example 3: Multi-dimensional jump offsets")
    print("-" * 50)
    
    n_timesteps = 15
    n_dim = 2
    
    # Different jump offsets for each dimension
    jump_offsets = np.array([0.5, -0.3])  # Positive jump in dim 0, negative in dim 1
    
    # Define transition and observation functions
    def transition_function(state, noise):
        return state + noise
    
    def observation_function(state, noise):
        return state + noise
    
    # Create filter
    ukf = UnscentedKalmanFilter(
        transition_functions=transition_function,
        observation_functions=observation_function,
        transition_covariance=0.1 * np.eye(n_dim),
        observation_covariance=0.1 * np.eye(n_dim),
        initial_state_mean=np.zeros(n_dim),
        initial_state_covariance=np.eye(n_dim),
        jump_offsets=jump_offsets,
        random_state=42,
    )
    
    # Sample and filter
    states, observations = ukf.sample(n_timesteps)
    filtered_means, _ = ukf.filter(observations)
    
    print(f"Initial state: [{states[0, 0]:.3f}, {states[0, 1]:.3f}]")
    print(f"Final state: [{states[-1, 0]:.3f}, {states[-1, 1]:.3f}]")
    print(f"Expected change: [{jump_offsets[0] * (n_timesteps-1):.3f}, "
          f"{jump_offsets[1] * (n_timesteps-1):.3f}]")
    print(f"Actual change: [{states[-1, 0] - states[0, 0]:.3f}, "
          f"{states[-1, 1] - states[0, 1]:.3f}]")
    print()


if __name__ == "__main__":
    example_1_constant_jump()
    example_2_time_varying_jumps()
    example_3_multidimensional()
