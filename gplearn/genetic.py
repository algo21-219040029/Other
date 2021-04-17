from abc import ABCMeta, abstractmethod
from sklearn.utils.validation import check_X_y
from sklearn.base import BaseEstimator, check_array, ClassifierMixin

class BaseSymbolic(BaseEstimator, metaclass=ABCMeta):
    """Base class for symbolic regression / classification estimators"""
    @abstractmethod
    def __init__(self,
                 population_size=1000):
        pass

    def fit(self, X, y, sample_weight=None) -> object:
        """

        Parameters
        ----------
        X
        y
        sample_weight

        Returns
        -------

        """
        # 检查sample_weight，确保sample_weight仅包含有限值。如果sample_weight的dtype类型是object，则强制转化为float。
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False)

        # 如果self是分类器，则不需要要求y是数值类型
        if isinstance(self, ClassifierMixin):
            X, y = check_X_y(X, y, numeric=False)

        params = self.get_params()

        # for gen in range(prior_generations, self.generations)




