from obp.utils import check_array, check_scalar, sample_action_fast

import numpy as np
from typing import Union
import scipy

from obp.dataset import logistic_reward_function

from sklearn.linear_model import LogisticRegression


class Policy:
    
    def __init__(self, n_actions, model=None, bandit_feedback=None, beta=1, len_list=1, random_state=42):
        self.n_actions = n_actions
        self.model = model
        self.bandit_feedback = bandit_feedback
        self.beta = beta
        self.len_list = len_list
        self.random_state = random_state
    
    def predict(self, context: np.ndarray) -> np.ndarray:
        """Predict best actions for new data.

        Note
        --------
        Action set predicted by this `predict` method can contain duplicate items.
        If a non-repetitive action set is needed, please use the `sample_action` method.

        Parameters
        -----------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Context vectors for new data.

        Returns
        -----------
        action_dist: array-like, shape (n_rounds_of_new_data, n_actions, len_list)
            Action choices made by a classifier, which can contain duplicate items.
            If a non-repetitive action set is needed, please use the `sample_action` method.

        """
        check_array(array=context, name="context", expected_dim=2)

        action_dist = self.predict_proba(context)

        action_dist_deterministic = np.zeros(action_dist.shape)
        action_dist_deterministic[np.arange(action_dist.shape[0]), np.argmax(action_dist, axis=1).flatten(), 0] = 1
        return action_dist_deterministic
    
    def sample(self, context):
        check_array(array=context, name="context", expected_dim=2)

        action_dist = self.predict_proba(context)
        actions = sample_action_fast(action_dist[:,:,0], self.random_state)

        action_dist_deterministic = np.zeros(action_dist.shape)
        action_dist_deterministic[np.arange(action_dist.shape[0]), actions, 0] = 1
        return action_dist_deterministic


    def predict_score(self, context: np.ndarray) -> np.ndarray:
        """Predict non-negative scores for all possible pairs of actions and positions.

        Parameters
        -----------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Context vectors for new data.

        Returns
        -----------
        score_predicted: array-like, shape (n_rounds_of_new_data, n_actions, len_list)
            Scores for all possible pairs of actions and positions predicted by a classifier.

        """
        check_array(array=context, name="context", expected_dim=2)

        n = context.shape[0]
        score_predicted = np.zeros((n, self.n_actions, self.len_list))
        for p in np.arange(self.len_list):
            if self.model is not None:
                score_predicteds_at_position = self.model.predict_proba(
                    context
                )[:,:,p]
            else:
                score_predicteds_at_position = self.bandit_feedback['expected_reward']
            score_predicted[:, :, p] = score_predicteds_at_position
        return score_predicted * self.beta
    
    def predict_proba(
        self,
        context: np.ndarray,
        action_context=None,
        random_state=None,
        tau: Union[int, float] = 1.0,
    ) -> np.ndarray:
        """Obtains action choice probabilities for new data based on scores predicted by a classifier.

        Note
        --------
        This `predict_proba` method obtains action choice probabilities for new data :math:`x \\in \\mathcal{X}`
        by applying the softmax function as follows:

        .. math::

            P (A = a | x) = \\frac{\\mathrm{exp}(f(x,a) / \\tau)}{\\sum_{a^{\\prime} \\in \\mathcal{A}} \\mathrm{exp}(f(x,a^{\\prime}) / \\tau)},

        where :math:`A` is a random variable representing an action, and :math:`\\tau` is a temperature hyperparameter.
        :math:`f: \\mathcal{X} \\times \\mathcal{A} \\rightarrow \\mathbb{R}_{+}`
        is a scoring function which is now implemented in the `predict_score` method.

        **Note that this method can be used only when `len_list=1`, please use the `sample_action` method otherwise.**

        Parameters
        ----------------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Context vectors for new data.

        tau: int or float, default=1.0
            A temperature parameter that controls the randomness of the action choice
            by scaling the scores before applying softmax.
            As :math:`\\tau \\rightarrow \\infty`, the algorithm will select arms uniformly at random.

        Returns
        -----------
        choice_prob: array-like, shape (n_rounds_of_new_data, n_actions, len_list)
            Action choice probabilities obtained by a trained classifier.

        """
        assert (
            self.len_list == 1
        ), "predict_proba method cannot be used when `len_list != 1`"
        check_array(array=context, name="context", expected_dim=2)
        check_scalar(tau, name="tau", target_type=(int, float), min_val=0)

        score_predicted = self.predict_score(context=context)
        choice_prob = scipy.special.softmax(score_predicted / tau, axis=1)
        return choice_prob #[:,:,0]


def get_bandit_field(bandit_data, fields, ideal):

    out = []
    for field in fields:
        field_obs = field + '_obs'

        if (not ideal) and (field_obs in bandit_data.keys()):
            field = field_obs
            
        out.append(bandit_data[field])

    return out



def softmax_from_reward(bandit_data, beta, tau=1, ideal=False, random_state=42):

    context, action_context = get_bandit_field(bandit_data, 
                                               ['context', 'action_context'],
                                               ideal)
    expected_reward = logistic_reward_function(context, action_context)
    pol = scipy.special.softmax((expected_reward * beta) / tau, axis=1)
    return pol[:, :, np.newaxis]


def greedy(bandit_data, beta, uniform=True, ideal=False, random_state=42):

    context, action_context = get_bandit_field(bandit_data, 
                                               ['context', 'action_context'],
                                               ideal)

    expected_reward = logistic_reward_function(context, action_context)

    base_pol = np.zeros_like(expected_reward)
    a = np.argmax(expected_reward, axis=1)

    base_pol[
        np.arange(expected_reward.shape[0]),
        a,
    ] = 1
    pol = (1.0 - beta) * base_pol
    pol += beta / expected_reward.shape[1]

    return pol[:, :, np.newaxis]


def content_based_filtering(bandit_data, ideal=False, random_state=42):

    context, action_context, action_embed, reward = get_bandit_field(
        bandit_data, 
        ['context', 'action_context', 'action_embed', 'reward'],
        ideal)

    model = LogisticRegression()

    X = np.hstack([context, action_embed])

    model.fit(X, reward)

    pol = np.zeros((len(context), len(action_context)))
    for i, con in enumerate(context):
        con = np.array([con,] * len(action_context))
        x = np.hstack([con, action_context])
        score = model.predict_proba(x)[:, 1]
        pol[i] = score / sum(score)

    return pol[:, :, np.newaxis]


