"""
Tests for jump-like inputs (Dirac Delta functions) support in unscented Kalman filters.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from pykalman import UnscentedKalmanFilter, AdditiveUnscentedKalmanFilter


class TestUnscentedJumpOffsets:
    """Test suite for jump offset functionality in unscented Kalman filters."""

    def test_unscented_filter_with_constant_jump(self):
        """Test UnscentedKalmanFilter filtering with constant jump offsets."""
        n_timesteps = 10
        n_dim_state = 1
        n_dim_obs = 1
        
        # System parameters
        transition_covariance = np.array([[0.1]])
        observation_covariance = np.array([[0.1]])
        initial_state_mean = np.array([0.0])
        initial_state_covariance = np.array([[1.0]])
        
        # Constant jump of 0.5 applied at each transition
        jump_offsets = np.array([0.5])
        
        # Generate observations
        np.random.seed(42)
        observations = np.random.randn(n_timesteps, n_dim_obs)
        
        # Define transition and observation functions
        def transition_function(state, noise):
            return state + noise
        
        def observation_function(state, noise):
            return state + noise
        
        # Create Unscented Kalman filter with jumps
        ukf_with_jumps = UnscentedKalmanFilter(
            transition_functions=transition_function,
            observation_functions=observation_function,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            jump_offsets=jump_offsets,
        )
        
        # Filter with jumps
        filtered_state_means_with_jumps, _ = ukf_with_jumps.filter(observations)
        
        # Create Unscented Kalman filter without jumps
        ukf_no_jumps = UnscentedKalmanFilter(
            transition_functions=transition_function,
            observation_functions=observation_function,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
        )
        
        # Filter without jumps
        filtered_state_means_no_jumps, _ = ukf_no_jumps.filter(observations)
        
        # The filtered states should be different
        assert not np.allclose(
            filtered_state_means_with_jumps, filtered_state_means_no_jumps
        ), "Jump offsets should affect filtering results"
        
        # With jumps, the filtered states should be higher (since we're adding 0.5 each time)
        assert np.mean(filtered_state_means_with_jumps) > np.mean(
            filtered_state_means_no_jumps
        ), "Positive jump offsets should increase state estimates"

    def test_unscented_filter_with_time_varying_jump(self):
        """Test UnscentedKalmanFilter filtering with time-varying jump offsets."""
        n_timesteps = 5
        n_dim_state = 1
        
        # System parameters
        transition_covariance = np.array([[0.01]])
        observation_covariance = np.array([[0.01]])
        initial_state_mean = np.array([0.0])
        initial_state_covariance = np.array([[1.0]])
        
        # Time-varying jumps: [0.0, 1.0, 0.0, -1.0]
        jump_offsets = np.array([[0.0], [1.0], [0.0], [-1.0]])
        
        # Simple observations
        observations = np.array([[0.1], [1.2], [1.1], [0.2], [0.0]])
        
        # Define transition and observation functions
        def transition_function(state, noise):
            return state + noise
        
        def observation_function(state, noise):
            return state + noise
        
        # Create Unscented Kalman filter with jumps
        ukf = UnscentedKalmanFilter(
            transition_functions=transition_function,
            observation_functions=observation_function,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            jump_offsets=jump_offsets,
        )
        
        # Filter
        filtered_state_means, filtered_state_covariances = ukf.filter(observations)
        
        # Check that filtering completes successfully
        assert filtered_state_means.shape == (n_timesteps, n_dim_state)
        assert filtered_state_covariances.shape == (n_timesteps, n_dim_state, n_dim_state)
        
        # Check that all covariances are positive definite
        for cov in filtered_state_covariances:
            assert np.all(np.linalg.eigvals(cov) > 0)

    def test_unscented_filter_with_multidimensional_jump(self):
        """Test UnscentedKalmanFilter filtering with multi-dimensional jump offsets."""
        n_timesteps = 5
        n_dim_state = 2
        n_dim_obs = 2
        
        # System parameters
        transition_covariance = 0.1 * np.eye(n_dim_state)
        observation_covariance = 0.1 * np.eye(n_dim_obs)
        initial_state_mean = np.zeros(n_dim_state)
        initial_state_covariance = np.eye(n_dim_state)
        
        # Jump offsets for 2D state: [0.5, -0.3] at each timestep
        jump_offsets = np.array([0.5, -0.3])
        
        # Random observations
        np.random.seed(42)
        observations = np.random.randn(n_timesteps, n_dim_obs)
        
        # Define transition and observation functions
        def transition_function(state, noise):
            return state + noise
        
        def observation_function(state, noise):
            return state + noise
        
        # Create Unscented Kalman filter with jumps
        ukf = UnscentedKalmanFilter(
            transition_functions=transition_function,
            observation_functions=observation_function,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            jump_offsets=jump_offsets,
        )
        
        # Filter
        filtered_state_means, filtered_state_covariances = ukf.filter(observations)
        
        # Check dimensions
        assert filtered_state_means.shape == (n_timesteps, n_dim_state)
        assert filtered_state_covariances.shape == (n_timesteps, n_dim_state, n_dim_state)
        
        # Create filter without jumps for comparison
        ukf_no_jumps = UnscentedKalmanFilter(
            transition_functions=transition_function,
            observation_functions=observation_function,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
        )
        filtered_no_jumps, _ = ukf_no_jumps.filter(observations)
        
        # The filtered states with jumps should trend higher in dim 0 and lower in dim 1
        assert np.mean(filtered_state_means[:, 0]) > np.mean(filtered_no_jumps[:, 0])
        assert np.mean(filtered_state_means[:, 1]) < np.mean(filtered_no_jumps[:, 1])

    def test_unscented_smooth_with_jump(self):
        """Test UnscentedKalmanFilter smoothing with jump offsets."""
        n_timesteps = 10
        
        # System parameters
        transition_covariance = np.array([[0.1]])
        observation_covariance = np.array([[0.1]])
        initial_state_mean = np.array([0.0])
        initial_state_covariance = np.array([[1.0]])
        jump_offsets = np.array([0.5])
        
        # Random observations
        np.random.seed(42)
        observations = np.random.randn(n_timesteps, 1)
        
        # Define transition and observation functions
        def transition_function(state, noise):
            return state + noise
        
        def observation_function(state, noise):
            return state + noise
        
        # Create Unscented Kalman filter with jumps
        ukf = UnscentedKalmanFilter(
            transition_functions=transition_function,
            observation_functions=observation_function,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            jump_offsets=jump_offsets,
        )
        
        # Smooth
        smoothed_state_means, smoothed_state_covariances = ukf.smooth(observations)
        
        # Check dimensions
        assert smoothed_state_means.shape == (n_timesteps, 1)
        assert smoothed_state_covariances.shape == (n_timesteps, 1, 1)
        
        # Smoothed covariances should be positive definite
        for cov in smoothed_state_covariances:
            assert np.all(np.linalg.eigvals(cov) > 0)

    def test_additive_filter_with_constant_jump(self):
        """Test AdditiveUnscentedKalmanFilter filtering with constant jump offsets."""
        n_timesteps = 10
        n_dim_state = 1
        n_dim_obs = 1
        
        # System parameters
        transition_covariance = np.array([[0.1]])
        observation_covariance = np.array([[0.1]])
        initial_state_mean = np.array([0.0])
        initial_state_covariance = np.array([[1.0]])
        
        # Constant jump of 0.5 applied at each transition
        jump_offsets = np.array([0.5])
        
        # Generate observations
        np.random.seed(42)
        observations = np.random.randn(n_timesteps, n_dim_obs)
        
        # Define transition and observation functions (additive noise)
        def transition_function(state):
            return state
        
        def observation_function(state):
            return state
        
        # Create Additive Unscented Kalman filter with jumps
        aukf_with_jumps = AdditiveUnscentedKalmanFilter(
            transition_functions=transition_function,
            observation_functions=observation_function,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            jump_offsets=jump_offsets,
        )
        
        # Filter with jumps
        filtered_state_means_with_jumps, _ = aukf_with_jumps.filter(observations)
        
        # Create Additive Unscented Kalman filter without jumps
        aukf_no_jumps = AdditiveUnscentedKalmanFilter(
            transition_functions=transition_function,
            observation_functions=observation_function,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
        )
        
        # Filter without jumps
        filtered_state_means_no_jumps, _ = aukf_no_jumps.filter(observations)
        
        # The filtered states should be different
        assert not np.allclose(
            filtered_state_means_with_jumps, filtered_state_means_no_jumps
        ), "Jump offsets should affect filtering results"
        
        # With jumps, the filtered states should be higher
        assert np.mean(filtered_state_means_with_jumps) > np.mean(
            filtered_state_means_no_jumps
        ), "Positive jump offsets should increase state estimates"

    def test_additive_smooth_with_jump(self):
        """Test AdditiveUnscentedKalmanFilter smoothing with jump offsets."""
        n_timesteps = 10
        
        # System parameters
        transition_covariance = np.array([[0.1]])
        observation_covariance = np.array([[0.1]])
        initial_state_mean = np.array([0.0])
        initial_state_covariance = np.array([[1.0]])
        jump_offsets = np.array([0.5])
        
        # Random observations
        np.random.seed(42)
        observations = np.random.randn(n_timesteps, 1)
        
        # Define transition and observation functions
        def transition_function(state):
            return state
        
        def observation_function(state):
            return state
        
        # Create Additive Unscented Kalman filter with jumps
        aukf = AdditiveUnscentedKalmanFilter(
            transition_functions=transition_function,
            observation_functions=observation_function,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            jump_offsets=jump_offsets,
        )
        
        # Smooth
        smoothed_state_means, smoothed_state_covariances = aukf.smooth(observations)
        
        # Check dimensions
        assert smoothed_state_means.shape == (n_timesteps, 1)
        assert smoothed_state_covariances.shape == (n_timesteps, 1, 1)
        
        # Smoothed covariances should be positive definite
        for cov in smoothed_state_covariances:
            assert np.all(np.linalg.eigvals(cov) > 0)

    def test_filter_with_none_jump(self):
        """Test that filtering works correctly when jump_offsets is None."""
        n_timesteps = 5
        
        # System parameters
        transition_covariance = np.array([[0.1]])
        observation_covariance = np.array([[0.1]])
        initial_state_mean = np.array([0.0])
        initial_state_covariance = np.array([[1.0]])
        
        observations = np.random.randn(n_timesteps, 1)
        
        # Define transition and observation functions
        def transition_function(state, noise):
            return state + noise
        
        def observation_function(state, noise):
            return state + noise
        
        # Create filter with jump_offsets=None (explicit)
        ukf_explicit_none = UnscentedKalmanFilter(
            transition_functions=transition_function,
            observation_functions=observation_function,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            jump_offsets=None,
        )
        
        # Create filter without jump_offsets parameter (implicit None)
        ukf_implicit_none = UnscentedKalmanFilter(
            transition_functions=transition_function,
            observation_functions=observation_function,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
        )
        
        # Both should produce the same results
        filtered_explicit, _ = ukf_explicit_none.filter(observations)
        filtered_implicit, _ = ukf_implicit_none.filter(observations)
        
        assert_array_almost_equal(filtered_explicit, filtered_implicit)

    def test_sample_with_jump(self):
        """Test sampling with jump offsets."""
        n_timesteps = 10
        n_dim_state = 1
        
        # System parameters
        transition_covariance = np.array([[0.1]])
        observation_covariance = np.array([[0.1]])
        initial_state_mean = np.array([0.0])
        initial_state_covariance = np.array([[1.0]])
        jump_offsets = np.array([0.5])
        
        # Define transition and observation functions
        def transition_function(state, noise):
            return state + noise
        
        def observation_function(state, noise):
            return state + noise
        
        # Create Unscented Kalman filter with jumps
        ukf = UnscentedKalmanFilter(
            transition_functions=transition_function,
            observation_functions=observation_function,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            jump_offsets=jump_offsets,
            random_state=42,
        )
        
        # Sample
        states, observations = ukf.sample(n_timesteps)
        
        # Check dimensions
        assert states.shape == (n_timesteps, n_dim_state)
        assert observations.shape == (n_timesteps, n_dim_state)
        
        # With positive jumps, states should generally increase
        state_differences = np.diff(states, axis=0)
        assert np.mean(state_differences) > 0.3  # Should be close to 0.5

    def test_filter_jump_offsets_parameter(self):
        """Test that jump_offsets can be passed as parameter to filter()."""
        n_timesteps = 10
        
        # System parameters
        transition_covariance = np.array([[0.1]])
        observation_covariance = np.array([[0.1]])
        initial_state_mean = np.array([0.0])
        initial_state_covariance = np.array([[1.0]])
        jump_offsets = np.array([0.5])
        
        # Generate observations
        np.random.seed(42)
        observations = np.random.randn(n_timesteps, 1)
        
        # Define transition and observation functions
        def transition_function(state, noise):
            return state + noise
        
        def observation_function(state, noise):
            return state + noise
        
        # Create UKF without jump_offsets in constructor
        ukf = UnscentedKalmanFilter(
            transition_functions=transition_function,
            observation_functions=observation_function,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
        )
        
        # Filter with jump_offsets passed as parameter
        filtered_with_param, _ = ukf.filter(observations, jump_offsets=jump_offsets)
        
        # Filter without jump_offsets
        filtered_without, _ = ukf.filter(observations)
        
        # Results should be different
        assert not np.allclose(filtered_with_param, filtered_without)
        
        # With positive jumps, states should be higher
        assert np.mean(filtered_with_param) > np.mean(filtered_without)

    def test_smooth_jump_offsets_parameter(self):
        """Test that jump_offsets can be passed as parameter to smooth()."""
        n_timesteps = 10
        
        # System parameters
        transition_covariance = np.array([[0.1]])
        observation_covariance = np.array([[0.1]])
        initial_state_mean = np.array([0.0])
        initial_state_covariance = np.array([[1.0]])
        jump_offsets = np.array([0.5])
        
        # Random observations
        np.random.seed(42)
        observations = np.random.randn(n_timesteps, 1)
        
        # Define transition and observation functions
        def transition_function(state, noise):
            return state + noise
        
        def observation_function(state, noise):
            return state + noise
        
        # Create UKF without jump_offsets in constructor
        ukf = UnscentedKalmanFilter(
            transition_functions=transition_function,
            observation_functions=observation_function,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
        )
        
        # Smooth with jump_offsets passed as parameter
        smoothed_with_param, _ = ukf.smooth(observations, jump_offsets=jump_offsets)
        
        # Smooth without jump_offsets
        smoothed_without, _ = ukf.smooth(observations)
        
        # Results should be different
        assert not np.allclose(smoothed_with_param, smoothed_without)
        
        # With positive jumps, states should be higher
        assert np.mean(smoothed_with_param) > np.mean(smoothed_without)

    def test_parameter_overrides_constructor(self):
        """Test that jump_offsets parameter overrides constructor value."""
        n_timesteps = 10
        
        # System parameters
        transition_covariance = np.array([[0.1]])
        observation_covariance = np.array([[0.1]])
        initial_state_mean = np.array([0.0])
        initial_state_covariance = np.array([[1.0]])
        
        # Constructor jump offset
        jump_offsets_constructor = np.array([0.5])
        # Parameter jump offset (different and larger)
        jump_offsets_parameter = np.array([1.0])
        
        # Generate observations
        np.random.seed(42)
        observations = np.random.randn(n_timesteps, 1)
        
        # Define transition and observation functions
        def transition_function(state, noise):
            return state + noise
        
        def observation_function(state, noise):
            return state + noise
        
        # Create UKF with jump_offsets in constructor
        ukf = UnscentedKalmanFilter(
            transition_functions=transition_function,
            observation_functions=observation_function,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            jump_offsets=jump_offsets_constructor,
        )
        
        # Filter with constructor value
        filtered_constructor, _ = ukf.filter(observations)
        
        # Filter with parameter value (should override constructor)
        filtered_parameter, _ = ukf.filter(observations, jump_offsets=jump_offsets_parameter)
        
        # Results should be different
        assert not np.allclose(filtered_constructor, filtered_parameter)
        
        # Parameter has larger jumps, so states should be higher
        assert np.mean(filtered_parameter) > np.mean(filtered_constructor)

    def test_additive_filter_jump_offsets_parameter(self):
        """Test that jump_offsets can be passed as parameter to AdditiveUKF filter()."""
        n_timesteps = 10
        
        # System parameters
        transition_covariance = np.array([[0.1]])
        observation_covariance = np.array([[0.1]])
        initial_state_mean = np.array([0.0])
        initial_state_covariance = np.array([[1.0]])
        jump_offsets = np.array([0.5])
        
        # Generate observations
        np.random.seed(42)
        observations = np.random.randn(n_timesteps, 1)
        
        # Define transition and observation functions
        def transition_function(state):
            return state
        
        def observation_function(state):
            return state
        
        # Create Additive UKF without jump_offsets in constructor
        aukf = AdditiveUnscentedKalmanFilter(
            transition_functions=transition_function,
            observation_functions=observation_function,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
        )
        
        # Filter with jump_offsets passed as parameter
        filtered_with_param, _ = aukf.filter(observations, jump_offsets=jump_offsets)
        
        # Filter without jump_offsets
        filtered_without, _ = aukf.filter(observations)
        
        # Results should be different
        assert not np.allclose(filtered_with_param, filtered_without)
        
        # With positive jumps, states should be higher
        assert np.mean(filtered_with_param) > np.mean(filtered_without)

    def test_additive_smooth_jump_offsets_parameter(self):
        """Test that jump_offsets can be passed as parameter to AdditiveUKF smooth()."""
        n_timesteps = 10
        
        # System parameters
        transition_covariance = np.array([[0.1]])
        observation_covariance = np.array([[0.1]])
        initial_state_mean = np.array([0.0])
        initial_state_covariance = np.array([[1.0]])
        jump_offsets = np.array([0.5])
        
        # Random observations
        np.random.seed(42)
        observations = np.random.randn(n_timesteps, 1)
        
        # Define transition and observation functions
        def transition_function(state):
            return state
        
        def observation_function(state):
            return state
        
        # Create Additive UKF without jump_offsets in constructor
        aukf = AdditiveUnscentedKalmanFilter(
            transition_functions=transition_function,
            observation_functions=observation_function,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
        )
        
        # Smooth with jump_offsets passed as parameter
        smoothed_with_param, _ = aukf.smooth(observations, jump_offsets=jump_offsets)
        
        # Smooth without jump_offsets
        smoothed_without, _ = aukf.smooth(observations)
        
        # Results should be different
        assert not np.allclose(smoothed_with_param, smoothed_without)
        
        # With positive jumps, states should be higher
        assert np.mean(smoothed_with_param) > np.mean(smoothed_without)
