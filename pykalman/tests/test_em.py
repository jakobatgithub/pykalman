import numpy as np
import numpy.ma as ma
import pytest
from numpy.testing import assert_array_almost_equal

from ..standard import (
    _em_observation_matrix,
    _em_observation_covariance,
    _em_transition_matrix,
    _em_transition_covariance,
    _em_initial_state_mean,
    _em_initial_state_covariance,
    _em_transition_offset,
    _em_observation_offset,
)

@pytest.fixture
def em_data():
    T, n_state, n_obs = 5, 2, 3
    observations = ma.array(np.random.randn(T, n_obs))
    smoothed_state_means = np.random.randn(T, n_state)
    smoothed_state_covariances = np.stack([np.eye(n_state)] * T)
    pairwise_covariances = np.stack([np.eye(n_state)] * T)
    transition_matrices = np.stack([np.eye(n_state)] * (T - 1))
    observation_matrices = np.stack([np.eye(n_obs, n_state)] * T)
    transition_offsets = np.zeros((T - 1, n_state))
    observation_offsets = np.zeros((T, n_obs))
    return {
        "observations": observations,
        "smoothed_state_means": smoothed_state_means,
        "smoothed_state_covariances": smoothed_state_covariances,
        "pairwise_covariances": pairwise_covariances,
        "transition_matrices": transition_matrices,
        "observation_matrices": observation_matrices,
        "transition_offsets": transition_offsets,
        "observation_offsets": observation_offsets,
    }

def test_em_observation_matrix(em_data):
    C = _em_observation_matrix(
        em_data["observations"],
        em_data["observation_offsets"],
        em_data["smoothed_state_means"],
        em_data["smoothed_state_covariances"]
    )
    assert C.shape == (3, 2)

def test_em_observation_covariance(em_data):
    R = _em_observation_covariance(
        em_data["observations"],
        em_data["observation_offsets"],
        em_data["observation_matrices"],
        em_data["smoothed_state_means"],
        em_data["smoothed_state_covariances"]
    )
    assert R.shape == (3, 3)

def test_em_transition_matrix(em_data):
    A = _em_transition_matrix(
        em_data["transition_offsets"],
        em_data["smoothed_state_means"],
        em_data["smoothed_state_covariances"],
        em_data["pairwise_covariances"]
    )
    assert A.shape == (2, 2)

def test_em_transition_covariance(em_data):
    Q = _em_transition_covariance(
        em_data["transition_matrices"],
        em_data["transition_offsets"],
        em_data["smoothed_state_means"],
        em_data["smoothed_state_covariances"],
        em_data["pairwise_covariances"]
    )
    assert Q.shape == (2, 2)

def test_em_initial_state_mean(em_data):
    mu_0 = _em_initial_state_mean(em_data["smoothed_state_means"])
    assert_array_almost_equal(mu_0, em_data["smoothed_state_means"][0])

def test_em_initial_state_covariance(em_data):
    mu_0 = _em_initial_state_mean(em_data["smoothed_state_means"])
    Sigma_0 = _em_initial_state_covariance(
        mu_0,
        em_data["smoothed_state_means"],
        em_data["smoothed_state_covariances"]
    )
    assert Sigma_0.shape == (2, 2)

def test_em_transition_offset(em_data):
    b = _em_transition_offset(
        em_data["transition_matrices"],
        em_data["smoothed_state_means"]
    )
    assert b.shape == (2,)

def test_em_observation_offset(em_data):
    d = _em_observation_offset(
        em_data["observation_matrices"],
        em_data["smoothed_state_means"],
        em_data["observations"]
    )
    assert d.shape == (3,)
