from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np

from obp.ope.estimators import BaseOffPolicyEstimator
from obp.utils import check_array, check_ope_inputs, estimate_confidence_interval_by_bootstrap
from sklearn.utils import check_scalar


def compute_groundtruth_reward(bandit_feedback, action_dist):
    action = action_dist.astype(bool).reshape(bandit_feedback["expected_reward"].shape)
    return np.mean(bandit_feedback['expected_reward'][action])


@dataclass
class NaiveMethod(BaseOffPolicyEstimator):
    """Naive Method (NM).

    Note
    -------
    NM estimates the policy value of evaluation policy :math:`\\pi_e` as

    .. math::

        \hat{V}_{\mathrm{NM}} (\pi_e; \mathcal{D}) := \mathbb{E}_{n}[\mathbb{I}_{\pi_e(x_t)=a_t}r_t],

    where :math:`\\mathcal{D}=\\{(x_i,a_i,r_i)\\}_{i=1}^{n}` is logged bandit data with :math:`n` observations collected by
    behavior policy :math:`\\pi_b`. :math:`\\pi_e: \\mathcal{X} \\rightarrow \\mathcal{A}` is the function
    representing action choices by the evaluation policy realized during offline bandit simulation.
    :math:`\\mathbb{E}_{n}[\\cdot]` is the empirical average over :math:`n` observations in :math:`\\mathcal{D}`.

    Parameters
    ----------
    estimator_name: str, default='nm'.
        Name of the estimator.
    """

    estimator_name: str = "nm"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ------------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Estimated rewards for each observation.

        """
    
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        est_reward = action_dist[np.arange(action.shape[0]), action, position]
        return est_reward

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """Estimate the policy value of evaluation policy.

        Parameters
        ------------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        Returns
        ----------
        V_hat: float
            Estimated policy value of evaluation policy.

        """
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        check_ope_inputs(
            action_dist=action_dist, position=position, action=action, reward=reward
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            action_dist=action_dist,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        check_ope_inputs(
            action_dist=action_dist, position=position, action=action, reward=reward
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            action_dist=action_dist,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )
    

@dataclass
class MAP(BaseOffPolicyEstimator):
    """MAP (NM).

    Note
    -------
    MAP estimates the policy value of evaluation policy :math:`\\pi_e` as

    .. math::

        \hat{V}_{\mathrm{NM}} (\pi_e; \mathcal{D}) := \mathbb{E}_{n}[\mathbb{I}_{\pi_e(x_t)=a_t}r_t],

    where :math:`\\mathcal{D}=\\{(x_i,a_i,r_i)\\}_{i=1}^{n}` is logged bandit data with :math:`n` observations collected by
    behavior policy :math:`\\pi_b`. :math:`\\pi_e: \\mathcal{X} \\rightarrow \\mathcal{A}` is the function
    representing action choices by the evaluation policy realized during offline bandit simulation.
    :math:`\\mathbb{E}_{n}[\\cdot]` is the empirical average over :math:`n` observations in :math:`\\mathcal{D}`.

    Parameters
    ----------
    estimator_name: str, default='nm'.
        Name of the estimator.
    """

    estimator_name: str = "map"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ------------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Estimated rewards for each observation.

        """
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        action_match = np.array(
            action_dist[np.arange(action.shape[0]), action, position] == 1
        )
        estimated_rewards = np.zeros_like(action_match)
        if action_match.sum() > 0.0:
            estimated_rewards = action_match * reward / np.mean(reward)
        return estimated_rewards
    

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """Estimate the policy value of evaluation policy.

        Parameters
        ------------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        Returns
        ----------
        V_hat: float
            Estimated policy value of evaluation policy.

        """
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        check_ope_inputs(
            action_dist=action_dist, position=position, action=action, reward=reward
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            action_dist=action_dist,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        check_ope_inputs(
            action_dist=action_dist, position=position, action=action, reward=reward
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            action_dist=action_dist,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class ReplayProbabilistic(BaseOffPolicyEstimator):
    """Relpay Probabilistic Method (RP).

    Note
    -------
    RP estimates the policy value of evaluation policy :math:`\\pi_e` as

    .. math::

        \\hat{V}_{\\mathrm{RM}} (\\pi_e; \\mathcal{D}) :=
        \\frac{\\mathbb{E}_{n}[\\mathbb{I} \\{ \\pi_e(x_t)=a_t \\} r_t]}{\\mathbb{E}_{n}[\\mathbb{I} \\{ \\pi_e(x_t)=a_t \\}]},

    where :math:`\\mathcal{D}=\\{(x_i,a_i,r_i)\\}_{i=1}^{n}` is logged bandit data with :math:`n` observations collected by
    behavior policy :math:`\\pi_b`. :math:`\\pi_e: \\mathcal{X} \\rightarrow \\mathcal{A}` is the function
    representing action choices by the evaluation policy realized during offline bandit simulation.
    :math:`\\mathbb{E}_{n}[\\cdot]` is the empirical average over :math:`n` observations in :math:`\\mathcal{D}`.

    Parameters
    ----------
    estimator_name: str, default='rm'.
        Name of the estimator.

    References
    ------------
    Lihong Li, Wei Chu, John Langford, and Xuanhui Wang.
    "Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms.", 2011.

    """

    estimator_name: str = "rp"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ------------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Estimated rewards for each observation.

        """
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        action_dist_deterministic = np.zeros(action_dist.shape)
        action_dist_deterministic[np.arange(action_dist.shape[0]), np.argmax(action_dist, axis=1).flatten(), 0] = 1

        action_match = np.array(
            action_dist_deterministic[np.arange(action.shape[0]), action, position] == 1
        )
        estimated_rewards = np.zeros_like(action_match)
        if action_match.sum() > 0.0:
            estimated_rewards = action_match * reward / action_match.sum()
        return estimated_rewards

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """Estimate the policy value of evaluation policy.

        Parameters
        ------------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        Returns
        ----------
        V_hat: float
            Estimated policy value of evaluation policy.

        """
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        check_ope_inputs(
            action_dist=action_dist, position=position, action=action, reward=reward
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            action_dist=action_dist,
        ).sum()

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        check_ope_inputs(
            action_dist=action_dist, position=position, action=action, reward=reward
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            action_dist=action_dist,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )
    

@dataclass
class RECAP(BaseOffPolicyEstimator):
    """Balanced Inverse Probability Weighting (B-IPW) Estimator.

    Note
    -------
    B-IPW estimates the policy value of evaluation policy :math:`\\pi_e` as

    .. math::

        \\hat{V}_{\\mathrm{B-IPW}} (\\pi_e; \\mathcal{D}) := \\frac{\\mathbb{E}_{\\mathcal{D}} [\\hat{w}(x_i,a_i) r_i]}{\\mathbb{E}_{\\mathcal{D}} [\\hat{w}(x_i,a_i)},

    where :math:`\\mathcal{D}=\\{(x_i,a_i,r_t)\\}_{t=1}^{T}` is logged bandit data with :math:`n` observations collected by
    a behavior policy :math:`\\pi_b`.
    :math:`\\hat{w}(x,a):=\\Pr[C=1|x,a] / \\Pr[C=0|x,a]`, where :math:`\\Pr[C=1|x,a]` is the probability that the data coming from the evaluation policy given action :math:`a` and :math:`x`.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`n` observations in :math:`\\mathcal{D}`.
    When the clipping is applied, large importance weights are clipped as :math:`\\hat{w_c}(x,a) := \\min \\{ \\lambda, \\hat{w}(x,a) \\}`
    where :math:`\\lambda (>0)` is a hyperparameter to define a maximum value allowed for importance weights.

    B-IPW re-weights the rewards by the importance weights estimated via a supervised classification procedure, and thus can be used even when the behavior policy (or the propensity score of the behavior policy) is not known. `obp.ope.ImportanceWeightEstimator` can be used to estimate the importance weights for B-IPW.


    Note that, in the reference paper, B-IPW is defined as follows (only when the evaluation policy is deterministic):

    .. math::

        \\hat{V}_{\\mathrm{B-IPW}} (\\pi_e; \\mathcal{D}) := \\frac{\\mathbb{E}_{\\mathcal{D}} [ \\hat{w}(x_t,\\pi_e (x_t)) r_i]}{\\mathbb{E}_{\\mathcal{D}} [ \\hat{w}(x_t,\\pi_e (x_t))},

    where :math:`\\pi_e` is a deterministic evaluation policy. We modify this original definition to adjust to stochastic evaluation policies.

    Parameters
    ------------
    lambda_: float, default=np.inf
        A maximum possible value of the importance weight.
        When a positive finite value is given, importance weights larger than `lambda_` will be clipped.

    estimator_name: str, default='b-ipw'.
        Name of the estimator.

    References
    ------------
    Arjun Sondhi, David Arbour, and Drew Dimmery
    "Balanced Off-Policy Evaluation in General Action Spaces.", 2020.

    """

    lambda_: float = np.inf
    estimator_name: str = "recap"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(
            self.lambda_,
            name="lambda_",
            target_type=(int, float),
            min_val=0.0,
        )
        if self.lambda_ != self.lambda_:
            raise ValueError("`lambda_` must not be nan")

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        estimated_importance_weights: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        estimated_importance_weights: array-like, shape (n_rounds,)
            Importance weights estimated via supervised classification using `obp.ope.ImportanceWeightEstimator`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Estimated rewards for each observation.

        """
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        rr = np.minimum(action_dist[np.arange(action.shape[0]), action, position], 0.05)
        rr = action_dist[np.arange(action.shape[0]), action, position]

        iw = estimated_importance_weights * rr
        # weight clipping
        if isinstance(iw, np.ndarray):
            iw = np.minimum(iw, self.lambda_)
        return reward * iw / iw.mean()

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        estimated_importance_weights: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the policy value of evaluation policy.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        estimated_importance_weights: array-like, shape (n_rounds,)
            Importance weights estimated via supervised classification using `obp.ope.ImportanceWeightEstimator`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        Returns
        ----------
        V_hat: float
            Estimated policy value of evaluation policy.

        """
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        check_array(
            array=estimated_importance_weights,
            name="estimated_importance_weights",
            expected_dim=1,
        )
        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            estimated_importance_weights=estimated_importance_weights,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)
        return self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            estimated_importance_weights=estimated_importance_weights,
            action_dist=action_dist,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        estimated_importance_weights: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        estimated_importance_weights: array-like, shape (n_rounds,)
            Importance weights estimated via supervised classification using `obp.ope.ImportanceWeightEstimator`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        check_array(
            array=estimated_importance_weights,
            name="estimated_importance_weights",
            expected_dim=1,
        )
        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            estimated_importance_weights=estimated_importance_weights,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            estimated_importance_weights=estimated_importance_weights,
            action_dist=action_dist,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )