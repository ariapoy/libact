"""K-Center-Greedy
Documentation.
Reference.
"""
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from scipy.spatial.distance import cdist

from libact.base.interfaces import QueryStrategy, ContinuousModel, \
    ProbabilisticModel
from libact.utils import inherit_docstring_from, zip, seed_random_state


class KCenterGreedy(QueryStrategy):

    """K-Center-Greedy

    This class implements K-Center-Greedy active learning algorithm [1]_.

    Parameters
    ----------
    transformer: :py:class:`An sklearn estimator supporting transform and/or fit_transform` object instance
        The base model used for training.


    References
    ----------

    .. [1] Core-Set... 
    """

    def __init__(self, *args, **kwargs):
        super(KCenterGreedy, self).__init__(*args, **kwargs)

        self.transformer = kwargs.pop('transformer', None)
        if self.transformer is None:
            self.transformer = FunctionTransformer()
        if not hasattr(self.transformer, "transform"):
            raise TypeError(
                "transformer has method: .transform()"
            )
        
        # initialize the transformer on labeled pool
        # Poy: We don't need it.
        # self.transformer.fit(self.dataset.X)

    def make_query(self, n=1):
        """Return the index of the sample to be queried and labeled and
        selection score of each sample. Read-only.

        No modification to the internal states.

        Returns
        -------
        ask_ids : list
            The batch of indexes of the next unlabeled samples to be queried and labeled.

        """
        dataset = self.dataset
        # Train CNNs (models) from scratch (retrain) after each iteration [1]_.
        X_lbl_curr, y_lbl_curr = dataset.get_labeled_entries()
        idx_lbl_mask = dataset.get_labeled_mask()
        X = dataset._X
        self.transformer.fit(X_lbl_curr, y_lbl_curr)
        embed = self.transformer.transform(X)
        # Reference. KH Huang
        # https://github.com/ariapoy/deep-active-learning/blob/master/query_strategies/kcenter_greedy.py#L15
        # embed_label = embed[idx_lbl_mask]
        # embed_unlabel = embed[~idx_lbl_mask]
        # dist_mat = cdist(embed_unlabel, embed_label, metric="euclidean")
        dist_mat = cdist(embed, embed, metric="euclidean")
        dist_mat_ublxlbl = dist_mat[~idx_lbl_mask, :][:, idx_lbl_mask]

        # scores: min_{j \in s}, (s: label pool)
        res = []
        for b in range(n):
            scores = np.min(dist_mat_ublxlbl, axis=1)
            ask_id_pos = np.argmax(scores)
            unlabeled_entry_ids, _ = dataset.get_unlabeled_entries()
            ask_id = unlabeled_entry_ids[ask_id_pos]
            res.append(ask_id)
            # update dist_mat_ublxlbl
            # solve ckp2
            if idx_lbl_mask[ask_id] != True:
                idx_lbl_mask[ask_id] = True
            else:
                print("ind {0} in already selected".format(ask_id))
                continue

            dist_mat_ublxlbl = np.delete(dist_mat_ublxlbl, ask_id_pos, 0)
            dist_mat_ublxlbl = np.append(dist_mat_ublxlbl, dist_mat[~idx_lbl_mask, ask_id][:, None], axis=1)

        return res
