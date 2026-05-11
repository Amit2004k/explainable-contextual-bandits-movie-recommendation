"""
bandits.py
----------
Contextual Bandit algorithms for recommendation:
  - LinUCB (Disjoint)
  - Thompson Sampling (Linear)
"""

import numpy as np


class LinUCB:
    """
    Disjoint LinUCB contextual bandit (Li et al., 2010).

    Parameters
    ----------
    context_dim : int
        Dimension of context vector per arm
    alpha : float
        Exploration parameter (UCB width)
    """

    def __init__(self, context_dim: int, alpha: float = 1.0):
        self.context_dim = context_dim
        self.alpha = alpha
        self.A = {}   # A[arm] = (d x d) matrix
        self.b = {}   # b[arm] = (d,) vector

    def _init_arm(self, arm_id):
        if arm_id not in self.A:
            self.A[arm_id] = np.identity(self.context_dim)
            self.b[arm_id] = np.zeros(self.context_dim)

    def select(self, arm_contexts: dict) -> int:
        """
        Select the arm with highest UCB score.

        Parameters
        ----------
        arm_contexts : dict {arm_id: context_vector (d,)}

        Returns
        -------
        int : selected arm_id
        """
        ucb_scores = {}
        for arm_id, x in arm_contexts.items():
            self._init_arm(arm_id)
            A_inv = np.linalg.inv(self.A[arm_id])
            theta = A_inv @ self.b[arm_id]
            ucb = theta @ x + self.alpha * np.sqrt(x @ A_inv @ x)
            ucb_scores[arm_id] = ucb
        return max(ucb_scores, key=ucb_scores.get)

    def update(self, arm_id: int, context: np.ndarray, reward: float):
        """Update arm parameters after observing reward."""
        self._init_arm(arm_id)
        self.A[arm_id] += np.outer(context, context)
        self.b[arm_id] += reward * context

    def theta(self, arm_id: int) -> np.ndarray:
        """Return current weight vector for an arm (for SHAP explanation)."""
        self._init_arm(arm_id)
        return np.linalg.inv(self.A[arm_id]) @ self.b[arm_id]


class ThompsonSamplingLinear:
    """
    Linear Thompson Sampling contextual bandit.
    Maintains a Gaussian posterior over reward weights per arm.

    Parameters
    ----------
    context_dim : int
    v : float
        Posterior variance scale (exploration strength)
    """

    def __init__(self, context_dim: int, v: float = 0.1):
        self.context_dim = context_dim
        self.v = v
        self.B = {}    # precision matrix
        self.mu = {}   # posterior mean
        self.f = {}    # sufficient statistic

    def _init_arm(self, arm_id):
        if arm_id not in self.B:
            self.B[arm_id] = np.identity(self.context_dim)
            self.mu[arm_id] = np.zeros(self.context_dim)
            self.f[arm_id] = np.zeros(self.context_dim)

    def select(self, arm_contexts: dict) -> int:
        samples = {}
        for arm_id, x in arm_contexts.items():
            self._init_arm(arm_id)
            B_inv = np.linalg.inv(self.B[arm_id])
            theta_sample = np.random.multivariate_normal(
                self.mu[arm_id], self.v**2 * B_inv
            )
            samples[arm_id] = theta_sample @ x
        return max(samples, key=samples.get)

    def update(self, arm_id: int, context: np.ndarray, reward: float):
        self._init_arm(arm_id)
        self.B[arm_id] += np.outer(context, context)
        self.f[arm_id] += reward * context
        self.mu[arm_id] = np.linalg.inv(self.B[arm_id]) @ self.f[arm_id]
