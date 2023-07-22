"""epsilon-greedy strategy
Reference: https://github.com/google/active-learning/blob/master/sampling_methods/mixture_of_samplers.py
"""

from libact.base.interfaces import QueryStrategy
from libact.utils import inherit_docstring_from, seed_random_state, zip
import numpy as np
import copy
from .uncertainty_sampling import UncertaintySampling
from .random_sampling import RandomSampling

class EpsilonGreedy(QueryStrategy):
    """Samples according to hybrid of uncertainty sampling and random sampling with epsilon greedy algorithm.

    """

    def __init__(self, dataset, epsilon=0.1, schedule=None, **kwargs):
        super(EpsilonGreedy, self).__init__(dataset, **kwargs)
        self.epsilon = epsilon
        if schedule is not None:
            self.schedule, self.eps_T  = schedule.split('_')
            self.eps_T = float(self.eps_T)
        else:
            self.schedule = None

        self.cntRS = 0
        self.cntUS = 0
        self.n_ubl = dataset.len_unlabeled()
        self.eps_history = [self.epsilon]

        # random seed
        random_state = kwargs.pop('random_state', None)
        self.random_state_ = seed_random_state(random_state)
        # classifier instance
        self.model = kwargs.pop('model', None)
        if self.model is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'model'"
            )
        # align the datasets, model for all query strategies
        self.RS = RandomSampling(self.dataset, random_state=random_state)
        self.US = UncertaintySampling(self.dataset, method='sm', model=self.model)

    @inherit_docstring_from(QueryStrategy)
    def make_query(self, n=1):
        p = self.random_state_.random()
        if p < self.epsilon:
            res = self.RS.make_query()
            self.cntRS += 1
        else:
            res = self.US.make_query()
            self.cntUS += 1

        if self.schedule == 'Linear':
            self.epsilon = self.eps_history[-1] + (self.eps_T - self.eps_history[0])/(self.n_ubl - 0)*1
        elif self.schedule == 'Exponential':
            pass

        self.eps_history.append(self.epsilon)

        return res

