"""Tests for control inputs in Unscented Kalman Filter."""

import numpy as np
from numpy import ma
from numpy.testing import assert_array_almost_equal

from ..unscented import AdditiveUnscentedKalmanFilter, UnscentedKalmanFilter


def test_unscented_filter_with_inputs():
    """Test that UnscentedKalmanFilter correctly handles control inputs."""
    # Define a simple linear system with control input
    # x_{t+1} = A*x_t + B*u_t + noise
    A = np.array([[1, 0.1], [0, 1]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0]])
    
    def transition_function(state, input_val, noise):
        return A.dot(state) + B.dot(input_val) + noise
    
    def observation_function(state, noise):
        return C.dot(state) + noise
    
    # Initial state
    x0 = np.array([0, 0])
    P0 = np.eye(2)
    
    # Process and observation noise
    Q = np.eye(2) * 0.01
    R = np.array([[0.1]])
    
    # Create control inputs
    n_timesteps = 10
    U = np.ones((n_timesteps - 1, 1)) * 0.5  # constant control input
    
    # Create filter
    kf = UnscentedKalmanFilter(
        transition_function, observation_function,
        Q, R, x0, P0,
        transition_inputs=U
    )
    
    # Generate observations using the same control inputs
    states, observations = kf.sample(n_timesteps, x0, random_state=42, U=U)
    
    # Run filter with control inputs
    filtered_means, filtered_covs = kf.filter(observations, U=U)
    
    # Check that the shape is correct
    assert filtered_means.shape == (n_timesteps, 2)
    assert filtered_covs.shape == (n_timesteps, 2, 2)
    
    # Check that the filtered means track the true states reasonably well
    # (not perfect due to noise, but should be close)
    errors = np.abs(filtered_means - states)
    assert np.mean(errors) < 1.0  # Mean error should be reasonable


def test_additive_filter_with_inputs():
    """Test that AdditiveUnscentedKalmanFilter correctly handles control inputs."""
    # Define a simple linear system with control input
    # x_{t+1} = A*x_t + B*u_t + noise (additive)
    A = np.array([[1, 0.1], [0, 1]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0]])
    
    def transition_function(state, input_val):
        return A.dot(state) + B.dot(input_val)
    
    def observation_function(state):
        return C.dot(state)
    
    # Initial state
    x0 = np.array([0, 0])
    P0 = np.eye(2)
    
    # Process and observation noise
    Q = np.eye(2) * 0.01
    R = np.array([[0.1]])
    
    # Create control inputs
    n_timesteps = 10
    U = np.ones((n_timesteps - 1, 1)) * 0.5  # constant control input
    
    # Create filter
    kf = AdditiveUnscentedKalmanFilter(
        transition_function, observation_function,
        Q, R, x0, P0,
        transition_inputs=U
    )
    
    # Generate observations using the same control inputs
    states, observations = kf.sample(n_timesteps, x0, random_state=42, U=U)
    
    # Run filter with control inputs
    filtered_means, filtered_covs = kf.filter(observations, U=U)
    
    # Check that the shape is correct
    assert filtered_means.shape == (n_timesteps, 2)
    assert filtered_covs.shape == (n_timesteps, 2, 2)
    
    # Check that the filtered means track the true states reasonably well
    errors = np.abs(filtered_means - states)
    assert np.mean(errors) < 1.0


def test_unscented_filter_without_inputs():
    """Test that UnscentedKalmanFilter still works without control inputs."""
    # Define a simple system without control input
    A = np.array([[1, 0.1], [0, 1]])
    C = np.array([[1, 0]])
    
    def transition_function(state, noise):
        return A.dot(state) + noise
    
    def observation_function(state, noise):
        return C.dot(state) + noise
    
    x0 = np.array([0, 0])
    P0 = np.eye(2)
    Q = np.eye(2) * 0.01
    R = np.array([[0.1]])
    
    kf = UnscentedKalmanFilter(
        transition_function, observation_function,
        Q, R, x0, P0
    )
    
    # Generate and filter observations without inputs
    states, observations = kf.sample(10, x0, random_state=42)
    filtered_means, filtered_covs = kf.filter(observations)
    
    assert filtered_means.shape == (10, 2)
    assert filtered_covs.shape == (10, 2, 2)


def test_filter_update_with_inputs():
    """Test that filter_update works with control inputs."""
    A = np.array([[1, 0.1], [0, 1]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0]])
    
    def transition_function(state, input_val, noise):
        return A.dot(state) + B.dot(input_val) + noise
    
    def observation_function(state, noise):
        return C.dot(state) + noise
    
    x0 = np.array([0, 0])
    P0 = np.eye(2)
    Q = np.eye(2) * 0.01
    R = np.array([[0.1]])
    
    kf = UnscentedKalmanFilter(
        transition_function, observation_function,
        Q, R, x0, P0
    )
    
    # Test incremental filter update with control inputs
    u = np.array([0.5])
    observation = np.array([0.1])
    
    mean_next, cov_next = kf.filter_update(
        x0, P0, observation,
        transition_input=u
    )
    
    assert mean_next.shape == (2,)
    assert cov_next.shape == (2, 2)


def test_smooth_with_inputs():
    """Test that smoothing works with control inputs."""
    A = np.array([[1, 0.1], [0, 1]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0]])
    
    def transition_function(state, input_val, noise):
        return A.dot(state) + B.dot(input_val) + noise
    
    def observation_function(state, noise):
        return C.dot(state) + noise
    
    x0 = np.array([0, 0])
    P0 = np.eye(2)
    Q = np.eye(2) * 0.01
    R = np.array([[0.1]])
    
    n_timesteps = 10
    U = np.ones((n_timesteps - 1, 1)) * 0.5
    
    kf = UnscentedKalmanFilter(
        transition_function, observation_function,
        Q, R, x0, P0,
        transition_inputs=U
    )
    
    states, observations = kf.sample(n_timesteps, x0, random_state=42, U=U)
    
    # Test smoothing with control inputs
    smoothed_means, smoothed_covs = kf.smooth(observations, U=U)
    
    assert smoothed_means.shape == (n_timesteps, 2)
    assert smoothed_covs.shape == (n_timesteps, 2, 2)
    
    # Smoothed estimates should generally be better than filtered
    filtered_means, _ = kf.filter(observations, U=U)
    
    # Calculate errors
    filtered_errors = np.abs(filtered_means - states)
    smoothed_errors = np.abs(smoothed_means - states)
    
    # Smoothed should generally have lower mean error (though not guaranteed for every single point)
    # We just check that smoothing runs without error here
    assert smoothed_errors.mean() < 2.0  # Reasonable bound


def test_time_varying_inputs():
    """Test that time-varying control inputs work correctly."""
    A = np.array([[1, 0.1], [0, 1]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0]])
    
    def transition_function(state, input_val, noise):
        return A.dot(state) + B.dot(input_val) + noise
    
    def observation_function(state, noise):
        return C.dot(state) + noise
    
    x0 = np.array([0, 0])
    P0 = np.eye(2)
    Q = np.eye(2) * 0.01
    R = np.array([[0.1]])
    
    n_timesteps = 10
    # Create time-varying control inputs (sinusoidal)
    U = np.sin(np.arange(n_timesteps - 1)).reshape(-1, 1)
    
    kf = UnscentedKalmanFilter(
        transition_function, observation_function,
        Q, R, x0, P0
    )
    
    states, observations = kf.sample(n_timesteps, x0, random_state=42, U=U)
    filtered_means, _ = kf.filter(observations, U=U)
    
    assert filtered_means.shape == (n_timesteps, 2)
    
    # Verify that the states are affected by the time-varying inputs
    # The velocity should roughly follow the input pattern
    assert np.std(states[:, 1]) > 0.1  # velocity varies


def test_input_dimensions():
    """Test that multi-dimensional control inputs work correctly."""
    # 3D state, 2D control input
    A = np.eye(3)
    B = np.array([[1, 0], [0, 1], [0.5, 0.5]])
    C = np.array([[1, 0, 0], [0, 1, 0]])
    
    def transition_function(state, input_val, noise):
        return A.dot(state) + B.dot(input_val) + noise
    
    def observation_function(state, noise):
        return C.dot(state) + noise
    
    x0 = np.array([0, 0, 0])
    P0 = np.eye(3)
    Q = np.eye(3) * 0.01
    R = np.eye(2) * 0.1
    
    n_timesteps = 5
    U = np.random.randn(n_timesteps - 1, 2) * 0.1
    
    kf = UnscentedKalmanFilter(
        transition_function, observation_function,
        Q, R, x0, P0
    )
    
    states, observations = kf.sample(n_timesteps, x0, random_state=42, U=U)
    filtered_means, _ = kf.filter(observations, U=U)
    
    assert filtered_means.shape == (n_timesteps, 3)
    assert observations.shape == (n_timesteps, 2)


def test_additive_filter_update_with_inputs():
    """Test that additive filter_update works with control inputs."""
    A = np.array([[1, 0.1], [0, 1]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0]])
    
    def transition_function(state, input_val):
        return A.dot(state) + B.dot(input_val)
    
    def observation_function(state):
        return C.dot(state)
    
    x0 = np.array([0, 0])
    P0 = np.eye(2)
    Q = np.eye(2) * 0.01
    R = np.array([[0.1]])
    
    kf = AdditiveUnscentedKalmanFilter(
        transition_function, observation_function,
        Q, R, x0, P0
    )
    
    # Test incremental filter update with control inputs
    u = np.array([0.5])
    observation = np.array([0.1])
    
    mean_next, cov_next = kf.filter_update(
        x0, P0, observation,
        transition_input=u
    )
    
    assert mean_next.shape == (2,)
    assert cov_next.shape == (2, 2)


def test_inputs_in_constructor():
    """Test that control inputs can be set in the constructor and used later."""
    A = np.array([[1, 0.1], [0, 1]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0]])
    
    def transition_function(state, input_val, noise):
        return A.dot(state) + B.dot(input_val) + noise
    
    def observation_function(state, noise):
        return C.dot(state) + noise
    
    x0 = np.array([0, 0])
    P0 = np.eye(2)
    Q = np.eye(2) * 0.01
    R = np.array([[0.1]])
    
    n_timesteps = 10
    U = np.ones((n_timesteps - 1, 1)) * 0.5
    
    # Set inputs in constructor
    kf = UnscentedKalmanFilter(
        transition_function, observation_function,
        Q, R, x0, P0,
        transition_inputs=U
    )
    
    states, observations = kf.sample(n_timesteps, x0, random_state=42)
    
    # Filter without explicitly passing U (should use constructor value)
    filtered_means, _ = kf.filter(observations)
    
    assert filtered_means.shape == (n_timesteps, 2)
    
    # Compare with explicitly passing U - should be the same
    filtered_means2, _ = kf.filter(observations, U=U)
    assert_array_almost_equal(filtered_means, filtered_means2)
