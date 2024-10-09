from abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict
from typing import Optional

import numpy as np
from sklearn.utils import check_scalar

from obp.utils import check_array
from obp.utils import check_ope_inputs
from obp.utils import estimate_confidence_interval_by_bootstrap
from obp.ope.helper import estimate_bias_in_ope
from obp.ope.helper import estimate_high_probability_upper_bound_bias


@dataclass
class BaseOffPolicyEstimator(metaclass=ABCMeta):
    """Base class for OPE estimators."""

    @abstractmethod
    def _estimate_round_rewards(self) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards."""
        raise NotImplementedError

    @abstractmethod
    def estimate_policy_value(self) -> float:
        """Estimate the policy value of evaluation policy."""
        raise NotImplementedError

    @abstractmethod
    def estimate_interval(self) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap."""
        raise NotImplementedError
    

@dataclass
class NaiveEstimator(BaseOffPolicyEstimator):
    """Inverse Probability Weighting (IPW) Estimator.

    Note
    -------
    IPW estimates the policy value of evaluation policy :math:`\\pi_e` as

    .. math::

        \\hat{V}_{\\mathrm{IPW}} (\\pi_e; \\mathcal{D}) := \\mathbb{E}_{n} [ w(x_i,a_i) r_i],

    where :math:`\\mathcal{D}=\\{(x_i,a_i,r_i)\\}_{i=1}^{n}` is logged bandit data with :math:`n` observations collected by
    behavior policy :math:`\\pi_b`. :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\mathbb{E}_{n}[\\cdot]` is the empirical average over :math:`n` observations in :math:`\\mathcal{D}`.
    When the clipping is applied, a large importance weight is clipped as :math:`\\hat{w}(x,a) := \\min \\{ \\lambda, w(x,a) \\}`
    where :math:`\\lambda (>0)` is a hyperparameter to specify a maximum allowed importance weight.

    IPW re-weights the rewards by the ratio of the evaluation policy and behavior policy (importance weight).
    When the behavior policy is known, IPW is unbiased and consistent for the true policy value.
    However, it can have a large variance, especially when the evaluation policy significantly deviates from the behavior policy.

    Parameters
    ------------
    lambda_: float, default=np.inf
        A maximum possible value of the importance weight.
        When a positive finite value is given, importance weights larger than `lambda_` will be clipped.

    use_estimated_pscore: bool, default=False.
        If True, `estimated_pscore` is used, otherwise, `pscore` (the true propensity scores) is used.

    estimator_name: str, default='ipw'.
        Name of the estimator.

    References
    ------------
    Alex Strehl, John Langford, Lihong Li, and Sham M Kakade.
    "Learning from Logged Implicit Exploration Data"., 2010.

    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.

    """

    lambda_: float = np.inf
    use_estimated_pscore: bool = False
    estimator_name: str = "ipw"

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
        if not isinstance(self.use_estimated_pscore, bool):
            raise TypeError(
                f"`use_estimated_pscore` must be a bool, but {type(self.use_estimated_pscore)} is given"
            )

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
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

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

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

        iw = action_dist[np.arange(action.shape[0]), action, position] / pscore
        # weight clipping
        if isinstance(iw, np.ndarray):
            iw = np.minimum(iw, self.lambda_)
        return reward * iw

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        estimated_pscore: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the policy value of evaluation policy.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.
            If `use_estimated_pscore` is False, `pscore` must be given.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        estimated_pscore: array-like, shape (n_rounds,), default=None
            Estimated behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_b(a_i|x_i)`.
            If `self.use_estimated_pscore` is True, `estimated_pscore` must be given.

        Returns
        ----------
        V_hat: float
            Estimated policy value of evaluation policy.

        """
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        if self.use_estimated_pscore:
            check_array(array=estimated_pscore, name="estimated_pscore", expected_dim=1)
            pscore_ = estimated_pscore
        else:
            check_array(array=pscore, name="pscore", expected_dim=1)
            pscore_ = pscore

        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore_,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore_,
            action_dist=action_dist,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        estimated_pscore: Optional[np.ndarray] = None,
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

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.
            If `use_estimated_pscore` is False, `pscore` must be given.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        estimated_pscore: array-like, shape (n_rounds,), default=None
            Estimated behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_b(a_i|x_i)`.
            If `self.use_estimated_pscore` is True, `estimated_pscore` must be given.

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
        if self.use_estimated_pscore:
            check_array(array=estimated_pscore, name="estimated_pscore", expected_dim=1)
            pscore_ = estimated_pscore
        else:
            check_array(array=pscore, name="pscore", expected_dim=1)
            pscore_ = pscore

        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore_,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore_,
            action_dist=action_dist,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )

    def _estimate_mse_score(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        use_bias_upper_bound: bool = True,
        delta: float = 0.05,
        **kwargs,
    ) -> float:
        """Estimate the MSE score of a given clipping hyperparameter to conduct hyperparameter tuning.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        use_bias_upper_bound: bool, default=True
            Whether to use a bias upper bound in hyperparameter tuning.
            If False, the direct bias estimator is used to estimate the MSE. See Su et al.(2020) for details.

        delta: float, default=0.05
            A confidence delta to construct a high probability upper bound used in SLOPE.

        Returns
        ----------
        estimated_mse_score: float
            Estimated MSE score of a given clipping hyperparameter `lambda_`.
            MSE score is the sum of (high probability) upper bound of bias and the sample variance.
            This is estimated using the automatic hyperparameter tuning procedure
            based on Section 5 of Su et al.(2020).

        """
        n = reward.shape[0]
        # estimate the sample variance of IPW with clipping
        sample_variance = np.var(
            self._estimate_round_rewards(
                reward=reward,
                action=action,
                pscore=pscore,
                action_dist=action_dist,
                position=position,
            )
        )
        sample_variance /= n

        # estimate the (high probability) upper bound of the bias of IPW with clipping
        iw = action_dist[np.arange(n), action, position] / pscore
        if use_bias_upper_bound:
            bias_term = estimate_high_probability_upper_bound_bias(
                reward=reward, iw=iw, iw_hat=np.minimum(iw, self.lambda_), delta=delta
            )
        else:
            bias_term = estimate_bias_in_ope(
                reward=reward,
                iw=iw,
                iw_hat=np.minimum(iw, self.lambda_),
            )
        estimated_mse_score = sample_variance + (bias_term**2)

        return estimated_mse_score