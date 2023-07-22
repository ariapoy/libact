""" Uncertainty Sampling

smallest margin method (margin sampling).

"""
import numpy as np

from libact.base.interfaces import QueryStrategy, ContinuousModel, \
    ProbabilisticModel
from libact.utils import inherit_docstring_from, seed_random_state, zip


class USSoftmax(QueryStrategy):

    def __init__(self, dataset, tau=1, **kwargs):
        super(USSoftmax, self).__init__(dataset, **kwargs)
        self.tau = tau

        self.model = kwargs.pop('model', None)
        if self.model is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'model'"
            )

        self.model.train(self.dataset)

        self.method = kwargs.pop('method', 'sm')
        if self.method not in ['lc', 'sm', 'entropy', 'margin']:
            raise TypeError(
                "supported methods are ['lc', 'sm', 'entropy', 'margin'], the given one "
                "is: " + self.method
            )

        random_state = kwargs.pop('random_state', 1126)
        self.random_state_ = seed_random_state(random_state)

    def _get_scores(self):
        dataset = self.dataset
        self.model.train(dataset)
        unlabeled_entry_ids, X_pool = dataset.get_unlabeled_entries()

        if isinstance(self.model, ProbabilisticModel):
            dvalue = self.model.predict_proba(X_pool)
        elif isinstance(self.model, ContinuousModel):
            dvalue = self.model.predict_real(X_pool)

        if self.method == 'lc':  # least confident
            score = -np.max(dvalue, axis=1)

        elif self.method == 'sm':  # smallest margin
            if np.shape(dvalue)[1] > 2:
                # Find 2 largest decision values
                dvalue = -(np.partition(-dvalue, 2, axis=1)[:, :2])
            score = -np.abs(dvalue[:, 0] - dvalue[:, 1])

        elif self.method == 'entropy':
            score = np.sum(-dvalue * np.log(dvalue), axis=1)

        elif self.method == 'margin':
            # https://github.com/ariapoy/active-learning/blob/master/sampling_methods/margin_AL.py
            if len(dvalue.shape) < 2:
                min_margin = abs(dvalue)
            else:
                sort_distances = np.sort(dvalue, 1)[:, -2:]
                min_margin = sort_distances[:, 1] - sort_distances[:, 0]
            score = min_margin

        return zip(unlabeled_entry_ids, score)

    def softmax(self, qt, tau=1):
        """
        softmax function can be used to convert values into action probabilities
        """
        qt = np.array(qt)
        exp_qt_tau = np.exp(qt/tau)
        if np.sum(exp_qt_tau) == 0:  # prevent the (exp_qt_tau == 0).all()
            idmax = qt.argmax()
            exp_qt_tau[idmax] = 1

        Pt = exp_qt_tau / np.sum(exp_qt_tau)
        return Pt

    def make_query(self, n=1, return_score=False):
        dataset = self.dataset

        unlabeled_entry_ids, scores = zip(*self._get_scores())
        scores_prob = self.softmax(scores, tau=self.tau)
        # sample from the scores_prob
        ids = np.arange(len(unlabeled_entry_ids))
        ask_ids = self.random_state_.choice(ids, size=n, replace=False, p=scores_prob)

        res = [unlabeled_entry_ids[i] for i in ask_ids]
        return res
