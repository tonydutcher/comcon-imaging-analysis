import numpy as np
import pandas as pd
from copy import copy
from sklearn.base import BaseEstimator, ClassifierMixin

class RSAClassifier(BaseEstimator, ClassifierMixin):  
    def __init__(self, subject_dir, mask, item_labels, category_labels=None):
        self.metric = 'item'
        self._estimator_type = "regressor"
        self.subject_dir = subject_dir
        self.mask = mask
        self.item_labels = item_labels
        self.category_labels = category_labels

        if category_labels is not None:
            self.metric = 'category'

    def _get_item_category(self, item_id):
        return self.category_labels[self.item_labels==item_id]

    def fit(self, X, y=None):
        # X = neural data
        # y = object_id labels

        # create an "empty" localizer, fill it with input data
        from localiza import Localiza
        localizer = Localiza(self.subject_dir, self.mask)
        localizer.imgs = X
        localizer.features  = None
        localizer.nfeatures = X.shape[1]
        localizer.nframes   = X.shape[0]
        localizer.beta_labels = {}
        #localizer.label = {}


        # fill object_id field with labels
        localizer.beta_labels['object_id'] = np.array(y)
        #localizer.label['object_id'] = np.array(y)

        # create object_id set
        y_ids = set(y)
        localizer.object_ids = y_ids

        # item-item correlations
        corr = localizer.compare_objects_betas()
        #corr = localizer.compare_objects()

        #print "using", self.metric, "metric"
        if self.metric=='item':
            # lower the better!
            corr[corr==1] = np.nan
            scores = corr.mean()

        elif self.metric=='category':
            # column categories
            categories = np.array([self._get_item_category(item)[0] for item in corr.columns])
            #print "categories=", categories

            # within/between-category separation
            scores = {}
            for cat in set(list(categories)):
                is_cat = categories==cat
                within = corr.loc[is_cat,  is_cat]
                between= corr.loc[~is_cat, is_cat]
                # remove identity
                within[within==1] = np.nan
                # higher the better!
                scores[cat] = np.square(np.exp(within.median())-np.exp(between.median()))

            scores = pd.DataFrame(scores).transpose().mean().values
            #print "shape(scores)=", scores.shape

        # update estimator
        self.scores_ = scores

        return self

    def predict(self, X, y=None):
        try:
            getattr(self, "scores_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        return self.scores_

    def score(self, X, y=None):
        return np.mean(self.predict(X))

