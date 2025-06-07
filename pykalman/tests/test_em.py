import numpy as np
import numpy.ma as ma
import pytest
from numpy.testing import assert_array_almost_equal, assert_allclose

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


def test__em_observation_matrix_with_masked_data():
    n_timesteps, n_dim_obs, n_dim_state = 5, 3, 2

    np.random.seed(123)
    smoothed_state_means = np.random.randn(n_timesteps, n_dim_state)
    smoothed_state_covariances = np.stack([np.eye(n_dim_state) * 0.2 for _ in range(n_timesteps)])
    C_true = np.random.randn(n_dim_obs, n_dim_state)
    d = np.random.randn(n_dim_obs)

    Z = smoothed_state_means @ C_true.T + d + 0.01 * np.random.randn(n_timesteps, n_dim_obs)
    Z_masked = ma.array(Z, mask=np.zeros_like(Z, dtype=bool))
    Z_masked[1, 0] = ma.masked
    Z_masked[3, 2] = ma.masked

    C_est = _em_observation_matrix(Z_masked, d, smoothed_state_means, smoothed_state_covariances)

    assert C_est.shape == (n_dim_obs, n_dim_state)
    assert np.all(np.isfinite(C_est))


def test__em_observation_covariance_with_masked_data():
    n_timesteps, n_dim_obs, n_dim_state = 5, 3, 2

    np.random.seed(456)
    smoothed_state_means = np.random.randn(n_timesteps, n_dim_state)
    smoothed_state_covariances = np.stack([np.eye(n_dim_state) * 0.1 for _ in range(n_timesteps)])
    C = np.random.randn(n_timesteps, n_dim_obs, n_dim_state)
    d = np.random.randn(n_timesteps, n_dim_obs)
    Z = np.array([C[t] @ smoothed_state_means[t] + d[t] + 0.01 * np.random.randn(n_dim_obs) for t in range(n_timesteps)])

    Z_masked = ma.array(Z, mask=np.zeros_like(Z, dtype=bool))
    Z_masked[0, 1] = ma.masked
    Z_masked[4, 0] = ma.masked

    R_est = _em_observation_covariance(Z_masked, d, C, smoothed_state_means, smoothed_state_covariances)

    assert R_est.shape == (n_dim_obs, n_dim_obs)
    assert np.all(np.isfinite(R_est))
    assert np.all(np.diag(R_est) > 0)


def test__em_observation_offset_with_masked_data():
    n_timesteps = 5
    n_dim_obs = 3
    n_dim_state = 2

    np.random.seed(42)

    # True offset to recover
    d_true = np.array([1.0, -0.5, 2.0])

    # Fixed observation matrix C and state means
    observation_matrices = np.tile(np.array([[1.0, 0.0],
                                             [0.0, 1.0],
                                             [1.0, -1.0]]), (n_timesteps, 1, 1))

    smoothed_state_means = np.random.randn(n_timesteps, n_dim_state)

    # Observations: z_t = C_t x_t + d + noise
    noise = 0.01 * np.random.randn(n_timesteps, n_dim_obs)
    observations_data = np.einsum('tij,tj->ti', observation_matrices, smoothed_state_means) + d_true + noise

    # Mask some entries
    mask = np.zeros_like(observations_data, dtype=bool)
    mask[1, 0] = True  # mask z_0(1)
    mask[3, 2] = True  # mask z_2(3)
    observations = ma.array(observations_data, mask=mask)

    # Estimate offsets
    d_est = _em_observation_offset(observation_matrices, smoothed_state_means, observations)

    # Check shape and that the result is finite
    assert d_est.shape == (n_dim_obs,)
    assert np.all(np.isfinite(d_est))

    # Compare estimated vs. true offsets only for components with full observations
    for i in range(n_dim_obs):
        if not np.any(mask[:, i]):
            assert_allclose(d_est[i], d_true[i], atol=0.05)


def run_em_iterations(
    A_true, Q_true, C_true, R_true, d_true, mu_0_true, Sigma_0_true,
    T=50, mask_fraction=0.0, n_iter=10
):
    n_state = A_true.shape[0]
    n_obs = C_true.shape[0]

    x = np.zeros((T, n_state))
    z = np.zeros((T, n_obs))

    x[0] = np.random.multivariate_normal(mu_0_true, Sigma_0_true)
    for t in range(1, T):
        x[t] = A_true @ x[t - 1] + np.random.multivariate_normal(np.zeros(n_state), Q_true)
    for t in range(T):
        z[t] = C_true @ x[t] + d_true + np.random.multivariate_normal(np.zeros(n_obs), R_true)

    if mask_fraction > 0.0:
        mask = np.random.rand(*z.shape) < mask_fraction
        z = ma.array(z, mask=mask)

    A_est = np.eye(n_state)
    Q_est = np.eye(n_state)
    C_est = np.eye(n_obs, n_state)
    R_est = np.eye(n_obs)
    d_est = np.zeros(n_obs)
    mu_0_est = np.zeros(n_state)
    Sigma_0_est = np.eye(n_state)

    smoothed_means = x.copy()
    smoothed_covs = np.stack([np.eye(n_state) * 0.01] * T)
    pairwise_covs = np.stack([np.eye(n_state) * 0.01] * T)

    for _ in range(n_iter):
        A_est = _em_transition_matrix(np.zeros((T - 1, n_state)), smoothed_means, smoothed_covs, pairwise_covs)
        Q_est = _em_transition_covariance(np.tile(A_est, (T - 1, 1, 1)), np.zeros((T - 1, n_state)), smoothed_means, smoothed_covs, pairwise_covs)
        C_est = _em_observation_matrix(z, d_est, smoothed_means, smoothed_covs)
        R_est = _em_observation_covariance(z, d_est, np.tile(C_est, (T, 1, 1)), smoothed_means, smoothed_covs)
        d_est = _em_observation_offset(np.tile(C_est, (T, 1, 1)), smoothed_means, z)
        mu_0_est = _em_initial_state_mean(smoothed_means)
        Sigma_0_est = _em_initial_state_covariance(mu_0_est, smoothed_means, smoothed_covs)

    return A_est, Q_est, C_est, R_est, d_est, mu_0_est, Sigma_0_est


@pytest.mark.parametrize("masked", [False, True])
def test__em_transition_matrix_convergence(masked):
    # np.random.seed(1)
    A_true = np.array([[0.8, 0.2], [-0.1, 0.95]])
    Q_true = np.eye(2) * 0.01
    C_true = np.eye(3, 2)
    R_true = np.eye(3) * 0.02
    d_true = np.zeros(3)
    mu_0 = np.zeros(2)
    Sigma_0 = np.eye(2) * 0.1

    A_est, *_ = run_em_iterations(
        A_true, Q_true, C_true, R_true, d_true, mu_0, Sigma_0,
        T=100,
        mask_fraction=0.2 if masked else 0.0,
        n_iter=100
    )
    assert_allclose(A_est, A_true, atol=0.1)


@pytest.mark.parametrize("masked", [False, True])
def test__em_transition_covariance_convergence(masked):
    # np.random.seed(2)
    A_true = np.array([[0.9, 0.1], [-0.2, 0.95]])
    Q_true = np.array([[0.05, 0.01], [0.01, 0.02]])
    C_true = np.eye(3, 2)
    R_true = np.eye(3) * 0.02
    d_true = np.zeros(3)
    mu_0 = np.zeros(2)
    Sigma_0 = np.eye(2) * 0.1

    _, Q_est, *_ = run_em_iterations(
        A_true, Q_true, C_true, R_true, d_true, mu_0, Sigma_0,
        T=100,
        mask_fraction=0.2 if masked else 0.0,
        n_iter=50
    )
    assert_allclose(Q_est, Q_true, atol=0.1)


@pytest.mark.parametrize("masked", [False, True])
def test__em_observation_matrix_convergence(masked):
    # np.random.seed(3)
    A_true = np.eye(2)
    Q_true = np.eye(2) * 0.01
    C_true = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
    R_true = np.eye(3) * 0.01
    d_true = np.zeros(3)
    mu_0 = np.zeros(2)
    Sigma_0 = np.eye(2) * 0.1

    *_, C_est, _, _, _, _ = run_em_iterations(
        A_true, Q_true, C_true, R_true, d_true, mu_0, Sigma_0,
        T=100,
        mask_fraction=0.2 if masked else 0.0,
        n_iter=100
    )
    assert_allclose(C_est, C_true, atol=0.3)


@pytest.mark.parametrize("masked", [False, True])
def test__em_observation_covariance_convergence(masked):
    # np.random.seed(4)
    A_true = np.eye(2)
    Q_true = np.eye(2) * 0.01
    C_true = np.eye(3, 2)
    R_true = np.diag([0.01, 0.02, 0.03])
    d_true = np.zeros(3)
    mu_0 = np.zeros(2)
    Sigma_0 = np.eye(2) * 0.1

    *_, R_est, _, _, _ = run_em_iterations(
        A_true, Q_true, C_true, R_true, d_true, mu_0, Sigma_0,
        T=100,
        mask_fraction=0.2 if masked else 0.0,
        n_iter=100
    )
    assert_allclose(R_est, R_true, atol=0.1)


@pytest.mark.parametrize("masked", [False, True])
def test__em_observation_offset_convergence(masked):
    # np.random.seed(5)
    A_true = np.eye(2)
    Q_true = np.eye(2) * 0.01
    C_true = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
    R_true = np.eye(3) * 0.01
    d_true = np.array([1.0, -0.5, 2.0])
    mu_0 = np.zeros(2)
    Sigma_0 = np.eye(2) * 0.1

    *_, d_est, _, _ = run_em_iterations(
        A_true, Q_true, C_true, R_true, d_true, mu_0, Sigma_0,
        T=100,
        mask_fraction=0.2 if masked else 0.0,
        n_iter=100
    )
    assert_allclose(d_est, d_true, atol=0.1)


# def test__em_initial_state_mean_covariance_convergence():
#     # np.random.seed(6)
#     A_true = np.eye(2)
#     Q_true = np.eye(2)
#     C_true = np.eye(3, 2)
#     R_true = np.eye(3) * 0.02
#     d_true = np.zeros(3)
#     mu_0 = np.array([1.5, -2.0])
#     Sigma_0 = np.array([[1.0, 0.1], [0.1, 1.5]])

#     *_, mu_est, Sigma_est = run_em_iterations(
#         A_true, Q_true, C_true, R_true, d_true, mu_0, Sigma_0,
#         T=100,
#         mask_fraction=0.0,
#         n_iter=500
#     )
#     assert_allclose(mu_est, mu_0, atol=0.5)
#     assert_allclose(Sigma_est, Sigma_0, atol=0.02)
