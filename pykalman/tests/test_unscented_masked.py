"""Tests for masked observations in Unscented Kalman Filters."""

import numpy as np
from numpy import ma
from numpy.testing import assert_array_almost_equal

from ..unscented import AdditiveUnscentedKalmanFilter, UnscentedKalmanFilter


def test_unscented_fully_masked_observation():
    """Test UKF with fully masked observations."""
    # Build simple filter
    A = np.array([[1, 1], [0, 1]])
    C = np.array([[0.5, -0.3]])
    f = lambda x, y: A.dot(x) + y
    g = lambda x, y: C.dot(x) + y

    x = np.array([1, 1])
    P = np.array([[1, 0.1], [0.1, 1]])
    Q = np.eye(2) * 0.1
    R = 0.5

    kf = UnscentedKalmanFilter(f, g, Q, R, x, P, random_state=0)

    # Create observations with some fully masked
    Z = ma.array([0, 1, 2, 3], mask=[True, False, True, False])
    
    mu_filt, sigma_filt = kf.filter(Z)
    
    # Check that dimensions are correct
    assert mu_filt.shape == (4, 2)
    assert sigma_filt.shape == (4, 2, 2)
    
    # Check that masked observations don't cause issues
    assert np.all(np.isfinite(mu_filt))
    assert np.all(np.isfinite(sigma_filt))


def test_additive_fully_masked_observation():
    """Test additive UKF with fully masked observations."""
    # Build simple filter
    A = np.array([[1, 1], [0, 1]])
    C = np.array([[0.5, -0.3]])
    f = lambda x: A.dot(x)
    g = lambda x: C.dot(x)

    x = np.array([1, 1])
    P = np.array([[1, 0.1], [0.1, 1]])
    Q = np.eye(2) * 0.1
    R = 0.5

    kf = AdditiveUnscentedKalmanFilter(f, g, Q, R, x, P, random_state=0)

    # Create observations with some fully masked
    Z = ma.array([0, 1, 2, 3], mask=[True, False, True, False])
    
    mu_filt, sigma_filt = kf.filter(Z)
    
    # Check that dimensions are correct
    assert mu_filt.shape == (4, 2)
    assert sigma_filt.shape == (4, 2, 2)
    
    # Check that masked observations don't cause issues
    assert np.all(np.isfinite(mu_filt))
    assert np.all(np.isfinite(sigma_filt))


def test_unscented_partial_masked_observation():
    """Test UKF with partially masked multi-dimensional observations."""
    # Build filter with 2D observations
    A = np.array([[1, 1], [0, 1]])
    C = np.array([[0.5, -0.3], [0.2, 0.4]])  # 2D observation
    f = lambda x, y: A.dot(x) + y
    g = lambda x, y: C.dot(x) + y

    x = np.array([1, 1])
    P = np.array([[1, 0.1], [0.1, 1]])
    Q = np.eye(2) * 0.1
    R = np.eye(2) * 0.5

    kf = UnscentedKalmanFilter(f, g, Q, R, x, P, random_state=0)

    # Create 2D observations with partial masking
    Z = ma.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], 
                 mask=[[False, True], [False, False], [True, False]])
    
    mu_filt, sigma_filt = kf.filter(Z)
    
    # Check that dimensions are correct
    assert mu_filt.shape == (3, 2)
    assert sigma_filt.shape == (3, 2, 2)
    
    # Check that partially masked observations don't cause issues
    assert np.all(np.isfinite(mu_filt))
    assert np.all(np.isfinite(sigma_filt))


def test_additive_partial_masked_observation():
    """Test additive UKF with partially masked multi-dimensional observations."""
    # Build filter with 2D observations
    A = np.array([[1, 1], [0, 1]])
    C = np.array([[0.5, -0.3], [0.2, 0.4]])  # 2D observation
    f = lambda x: A.dot(x)
    g = lambda x: C.dot(x)

    x = np.array([1, 1])
    P = np.array([[1, 0.1], [0.1, 1]])
    Q = np.eye(2) * 0.1
    R = np.eye(2) * 0.5

    kf = AdditiveUnscentedKalmanFilter(f, g, Q, R, x, P, random_state=0)

    # Create 2D observations with partial masking
    Z = ma.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], 
                 mask=[[False, True], [False, False], [True, False]])
    
    mu_filt, sigma_filt = kf.filter(Z)
    
    # Check that dimensions are correct
    assert mu_filt.shape == (3, 2)
    assert sigma_filt.shape == (3, 2, 2)
    
    # Check that partially masked observations don't cause issues
    assert np.all(np.isfinite(mu_filt))
    assert np.all(np.isfinite(sigma_filt))


def test_unscented_filter_update_masked():
    """Test UKF filter_update with masked observations."""
    # Build simple filter
    A = np.array([[1, 1], [0, 1]])
    C = np.array([[0.5, -0.3]])
    f = lambda x, y: A.dot(x) + y
    g = lambda x, y: C.dot(x) + y

    x = np.array([1, 1])
    P = np.array([[1, 0.1], [0.1, 1]])
    Q = np.eye(2) * 0.1
    R = 0.5

    kf = UnscentedKalmanFilter(f, g, Q, R, x, P, random_state=0)

    # Test with None observation (should be treated as fully masked)
    mu_filt, sigma_filt = kf.filter_update(x, P, observation=None)
    
    # With no observation, state should just be predicted forward
    # (it won't be exactly x because of the transition, but should be finite)
    assert np.all(np.isfinite(mu_filt))
    assert np.all(np.isfinite(sigma_filt))
    
    # Test with masked observation
    obs_masked = ma.array([1.0], mask=[True])
    mu_filt2, sigma_filt2 = kf.filter_update(mu_filt, sigma_filt, observation=obs_masked)
    assert np.all(np.isfinite(mu_filt2))
    assert np.all(np.isfinite(sigma_filt2))


def test_additive_filter_update_masked():
    """Test additive UKF filter_update with masked observations."""
    # Build simple filter
    A = np.array([[1, 1], [0, 1]])
    C = np.array([[0.5, -0.3]])
    f = lambda x: A.dot(x)
    g = lambda x: C.dot(x)

    x = np.array([1, 1])
    P = np.array([[1, 0.1], [0.1, 1]])
    Q = np.eye(2) * 0.1
    R = 0.5

    kf = AdditiveUnscentedKalmanFilter(f, g, Q, R, x, P, random_state=0)

    # Test with None observation (should be treated as fully masked)
    mu_filt, sigma_filt = kf.filter_update(x, P, observation=None)
    
    # With no observation, state should just be predicted forward
    assert np.all(np.isfinite(mu_filt))
    assert np.all(np.isfinite(sigma_filt))
    
    # Test with masked observation
    obs_masked = ma.array([1.0], mask=[True])
    mu_filt2, sigma_filt2 = kf.filter_update(mu_filt, sigma_filt, observation=obs_masked)
    assert np.all(np.isfinite(mu_filt2))
    assert np.all(np.isfinite(sigma_filt2))


def test_unscented_smoother_with_masked():
    """Test UKF smoother with masked observations."""
    # Build simple filter
    A = np.array([[1, 1], [0, 1]])
    C = np.array([[0.5, -0.3]])
    f = lambda x, y: A.dot(x) + y
    g = lambda x, y: C.dot(x) + y

    x = np.array([1, 1])
    P = np.array([[1, 0.1], [0.1, 1]])
    Q = np.eye(2) * 0.1
    R = 0.5

    kf = UnscentedKalmanFilter(f, g, Q, R, x, P, random_state=0)

    # Create observations with some masked
    Z = ma.array([0, 1, 2, 3, 4], mask=[True, False, True, False, False])
    
    mu_smooth, sigma_smooth = kf.smooth(Z)
    
    # Check that dimensions are correct
    assert mu_smooth.shape == (5, 2)
    assert sigma_smooth.shape == (5, 2, 2)
    
    # Check that masked observations don't cause issues
    assert np.all(np.isfinite(mu_smooth))
    assert np.all(np.isfinite(sigma_smooth))


def test_additive_smoother_with_masked():
    """Test additive UKF smoother with masked observations."""
    # Build simple filter
    A = np.array([[1, 1], [0, 1]])
    C = np.array([[0.5, -0.3]])
    f = lambda x: A.dot(x)
    g = lambda x: C.dot(x)

    x = np.array([1, 1])
    P = np.array([[1, 0.1], [0.1, 1]])
    Q = np.eye(2) * 0.1
    R = 0.5

    kf = AdditiveUnscentedKalmanFilter(f, g, Q, R, x, P, random_state=0)

    # Create observations with some masked
    Z = ma.array([0, 1, 2, 3, 4], mask=[True, False, True, False, False])
    
    mu_smooth, sigma_smooth = kf.smooth(Z)
    
    # Check that dimensions are correct
    assert mu_smooth.shape == (5, 2)
    assert sigma_smooth.shape == (5, 2, 2)
    
    # Check that masked observations don't cause issues
    assert np.all(np.isfinite(mu_smooth))
    assert np.all(np.isfinite(sigma_smooth))


def test_consistency_with_without_masking():
    """Test that unmasked observations give same results as non-masked."""
    # Build simple filter
    A = np.array([[1, 1], [0, 1]])
    C = np.array([[0.5, -0.3]])
    f = lambda x: A.dot(x)
    g = lambda x: C.dot(x)

    x = np.array([1, 1])
    P = np.array([[1, 0.1], [0.1, 1]])
    Q = np.eye(2) * 0.1
    R = 0.5

    kf = AdditiveUnscentedKalmanFilter(f, g, Q, R, x, P, random_state=0)

    # Create observations without mask
    Z_no_mask = np.array([0, 1, 2, 3])
    
    # Create same observations with all-False mask
    Z_with_mask = ma.array([0, 1, 2, 3], mask=[False, False, False, False])
    
    # Filter with both
    mu_filt_no_mask, sigma_filt_no_mask = kf.filter(Z_no_mask)
    mu_filt_with_mask, sigma_filt_with_mask = kf.filter(Z_with_mask)
    
    # Results should be the same
    assert_array_almost_equal(mu_filt_no_mask, mu_filt_with_mask)
    assert_array_almost_equal(sigma_filt_no_mask, sigma_filt_with_mask)
