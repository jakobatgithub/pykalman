"""
Tests for jump-like inputs (Dirac Delta functions) support in pykalman.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from pykalman import KalmanFilter


class TestJumpOffsets:
    """Test suite for jump offset functionality."""

    def test_filter_with_constant_jump(self):
        """Test filtering with constant jump offsets at each timestep."""
        # Setup a simple 1D system
        n_timesteps = 10
        n_dim_state = 1
        n_dim_obs = 1
        
        # System parameters
        transition_matrix = np.array([[1.0]])
        observation_matrix = np.array([[1.0]])
        transition_covariance = np.array([[0.1]])
        observation_covariance = np.array([[0.1]])
        initial_state_mean = np.array([0.0])
        initial_state_covariance = np.array([[1.0]])
        
        # Constant jump of 0.5 applied at each transition (constant, will be broadcast)
        jump_offsets = np.array([0.5])
        
        # Generate observations
        np.random.seed(42)
        true_state = [0.0]
        observations = []
        for t in range(n_timesteps):
            # State evolution with jump
            true_state = [true_state[0] + 0.5 + np.random.randn() * 0.316]
            obs = true_state[0] + np.random.randn() * 0.316
            observations.append([obs])
        
        observations = np.array(observations)
        
        # Create Kalman filter with jumps
        kf_with_jumps = KalmanFilter(
            transition_matrices=transition_matrix,
            observation_matrices=observation_matrix,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            jump_offsets=jump_offsets,
        )
        
        # Filter with jumps
        filtered_state_means_with_jumps, _ = kf_with_jumps.filter(observations)
        
        # Create Kalman filter without jumps for comparison
        kf_no_jumps = KalmanFilter(
            transition_matrices=transition_matrix,
            observation_matrices=observation_matrix,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
        )
        
        # Filter without jumps
        filtered_state_means_no_jumps, _ = kf_no_jumps.filter(observations)
        
        # The filtered states should be different
        assert not np.allclose(
            filtered_state_means_with_jumps, filtered_state_means_no_jumps
        ), "Jump offsets should affect filtering results"
        
        # With jumps, the filtered states should be higher (since we're adding 0.5 each time)
        assert np.mean(filtered_state_means_with_jumps) > np.mean(
            filtered_state_means_no_jumps
        ), "Positive jump offsets should increase state estimates"

    def test_filter_with_time_varying_jump(self):
        """Test filtering with time-varying jump offsets."""
        n_timesteps = 5
        n_dim_state = 1
        
        # System parameters
        transition_matrix = np.array([[1.0]])
        observation_matrix = np.array([[1.0]])
        transition_covariance = np.array([[0.01]])
        observation_covariance = np.array([[0.01]])
        initial_state_mean = np.array([0.0])
        initial_state_covariance = np.array([[1.0]])
        
        # Time-varying jumps: [0.0, 1.0, 0.0, -1.0] (n_timesteps-1 values)
        jump_offsets = np.array([[0.0], [1.0], [0.0], [-1.0]])
        
        # Simple observations
        observations = np.array([[0.1], [1.2], [1.1], [0.2], [0.0]])
        
        # Create Kalman filter with jumps
        kf = KalmanFilter(
            transition_matrices=transition_matrix,
            observation_matrices=observation_matrix,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            jump_offsets=jump_offsets,
        )
        
        # Filter
        filtered_state_means, filtered_state_covariances = kf.filter(observations)
        
        # Check that filtering completes successfully
        assert filtered_state_means.shape == (n_timesteps, n_dim_state)
        assert filtered_state_covariances.shape == (n_timesteps, n_dim_state, n_dim_state)
        
        # Check that all covariances are positive definite
        for cov in filtered_state_covariances:
            assert np.all(np.linalg.eigvals(cov) > 0)

    def test_filter_update_with_jump(self):
        """Test online filtering with jump offsets using batch filter."""
        # System parameters
        n_timesteps = 5
        transition_matrix = np.array([[1.0]])
        observation_matrix = np.array([[1.0]])
        transition_covariance = np.array([[0.1]])
        observation_covariance = np.array([[0.1]])
        initial_state_mean = np.array([0.0])
        initial_state_covariance = np.array([[1.0]])
        jump_offsets = np.array([0.5])  # Constant jump
        
        observations = np.array([[0.6], [1.2], [1.8], [2.4], [3.0]])
        
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
        
        # Filter without jumps
        kf_no_jumps = KalmanFilter(
            transition_matrices=transition_matrix,
            observation_matrices=observation_matrix,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
        )
        filtered_no_jumps, _ = kf_no_jumps.filter(observations)
        
        # The state estimates with jumps should be higher
        assert np.mean(filtered_with_jumps) > np.mean(filtered_no_jumps), \
            "Cumulative jump offsets should increase state estimates"

    def test_filter_with_none_jump(self):
        """Test that filtering works correctly when jump_offsets is None."""
        n_timesteps = 5
        
        # System parameters
        transition_matrix = np.array([[1.0]])
        observation_matrix = np.array([[1.0]])
        transition_covariance = np.array([[0.1]])
        observation_covariance = np.array([[0.1]])
        initial_state_mean = np.array([0.0])
        initial_state_covariance = np.array([[1.0]])
        
        observations = np.random.randn(n_timesteps, 1)
        
        # Create filter with jump_offsets=None (explicit)
        kf_explicit_none = KalmanFilter(
            transition_matrices=transition_matrix,
            observation_matrices=observation_matrix,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            jump_offsets=None,
        )
        
        # Create filter without jump_offsets parameter (implicit None)
        kf_implicit_none = KalmanFilter(
            transition_matrices=transition_matrix,
            observation_matrices=observation_matrix,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
        )
        
        # Both should produce the same results
        filtered_explicit, _ = kf_explicit_none.filter(observations)
        filtered_implicit, _ = kf_implicit_none.filter(observations)
        
        assert_array_almost_equal(filtered_explicit, filtered_implicit)

    def test_filter_with_multidimensional_jump(self):
        """Test filtering with multi-dimensional state and jumps."""
        n_timesteps = 5
        n_dim_state = 2
        n_dim_obs = 2
        
        # System parameters
        transition_matrix = np.eye(n_dim_state)
        observation_matrix = np.eye(n_dim_obs)
        transition_covariance = 0.1 * np.eye(n_dim_state)
        observation_covariance = 0.1 * np.eye(n_dim_obs)
        initial_state_mean = np.zeros(n_dim_state)
        initial_state_covariance = np.eye(n_dim_state)
        
        # Jump offsets for 2D state: [0.5, -0.3] at each timestep (constant)
        jump_offsets = np.array([0.5, -0.3])
        
        # Random observations
        np.random.seed(42)
        observations = np.random.randn(n_timesteps, n_dim_obs)
        
        # Create Kalman filter with jumps
        kf = KalmanFilter(
            transition_matrices=transition_matrix,
            observation_matrices=observation_matrix,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            jump_offsets=jump_offsets,
        )
        
        # Filter
        filtered_state_means, filtered_state_covariances = kf.filter(observations)
        
        # Check dimensions
        assert filtered_state_means.shape == (n_timesteps, n_dim_state)
        assert filtered_state_covariances.shape == (n_timesteps, n_dim_state, n_dim_state)
        
        # Over time, with positive jumps in dimension 0, the average should increase
        # and with negative jumps in dimension 1, the average should decrease  
        # (compared to no jumps). Check trend over multiple steps.
        # Create filter without jumps for comparison
        kf_no_jumps = KalmanFilter(
            transition_matrices=transition_matrix,
            observation_matrices=observation_matrix,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
        )
        filtered_no_jumps, _ = kf_no_jumps.filter(observations)
        
        # The filtered states with jumps should trend higher in dim 0 and lower in dim 1
        assert np.mean(filtered_state_means[:, 0]) > np.mean(filtered_no_jumps[:, 0])
        assert np.mean(filtered_state_means[:, 1]) < np.mean(filtered_no_jumps[:, 1])

    def test_smooth_with_jump(self):
        """Test smoothing with jump offsets."""
        n_timesteps = 10
        
        # System parameters
        transition_matrix = np.array([[1.0]])
        observation_matrix = np.array([[1.0]])
        transition_covariance = np.array([[0.1]])
        observation_covariance = np.array([[0.1]])
        initial_state_mean = np.array([0.0])
        initial_state_covariance = np.array([[1.0]])
        jump_offsets = np.array([0.5])
        
        # Random observations
        np.random.seed(42)
        observations = np.random.randn(n_timesteps, 1)
        
        # Create Kalman filter with jumps
        kf = KalmanFilter(
            transition_matrices=transition_matrix,
            observation_matrices=observation_matrix,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            jump_offsets=jump_offsets,
        )
        
        # Smooth
        smoothed_state_means, smoothed_state_covariances = kf.smooth(observations)
        
        # Check dimensions
        assert smoothed_state_means.shape == (n_timesteps, 1)
        assert smoothed_state_covariances.shape == (n_timesteps, 1, 1)
        
        # Smoothed covariances should be positive definite
        for cov in smoothed_state_covariances:
            assert np.all(np.linalg.eigvals(cov) > 0)
