import numpy as np

from typing import Callable

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import Normalizer

class FeatureCreatorPlaceholder(BaseEstimator, TransformerMixin):
    def __init__(self, n_features, new_dim, func: Callable = np.cos):
        self.n_features = n_features
        self.new_dim = new_dim
        self.w = None
        self.b = None
        self.func = func

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X


class RandomFeatureCreator(FeatureCreatorPlaceholder):
    def fit(self, X, y=None):
        # 2. Для полученной выборки оценить гиперпараметр \sigma^2 с помощью эвристики 
        # рекомендуем считать медиану не по всем парам объектов, а по случайному подмножеству из где-то миллиона пар объектов 
        ind =  np.random.choice(X.shape[0], size=(2, 1000))
        ind = ind[:, ind[0] != ind[1]]
        # \sigma^2 = median_{i, j = 1, \dots, \ell, i \neq j} {\sum_{k = 1}^{d} (x_{ik} - x_{jk})^2}
        t = np.sum((X[ind[0]] - X[ind[1]]) ** 2, axis=1)
        self.sigma2 = np.median(t)

        # 3. Сгенерировать n_features наборов весов $w_j$ и сдвигов $b_j$.
        self.w = np.random.normal(0, 1 / np.sqrt(self.sigma2), size=(X.shape[1], self.n_features))
        self.b = np.random.uniform(-np.pi, np.pi, self.n_features)

        return self

    def transform(self, X, y=None):
        X_new = self.func(X @ self.w + self.b)
        return X_new


class OrthogonalRandomFeatureCreator(RandomFeatureCreator):
    def fit(self, X, y=None):
        raise NotImplementedError


class RFFPipeline(BaseEstimator):
    """
    Пайплайн, делающий последовательно три шага:
        1. Применение PCA
        2. Применение RFF
        3. Применение классификатора
    """
    def __init__(
            self,
            n_features: int = 1000,
            new_dim: int = 50,
            use_PCA: bool = True,
            feature_creator_class=FeatureCreatorPlaceholder,
            classifier_class=LogisticRegression,
            classifier_params=None,
            func=np.cos,
    ):
        """
        :param n_features: Количество признаков, генерируемых RFF
        :param new_dim: Количество признаков, до которых сжимает PCA
        :param use_PCA: Использовать ли PCA
        :param feature_creator_class: Класс, создающий признаки, по умолчанию заглушка
        :param classifier_class: Класс классификатора
        :param classifier_params: Параметры, которыми инициализируется классификатор
        :param func: Функция, которую получает feature_creator при инициализации.
                     Если не хотите, можете не использовать этот параметр.
        """
        self.n_features = n_features
        self.new_dim = new_dim
        self.use_PCA = use_PCA
        if classifier_params is None:
            classifier_params = {"max_iter" : 1000}
        self.classifier_class = classifier_class
        self.classifier = classifier_class(**classifier_params)
        self.feature_creator = feature_creator_class(
            n_features=self.n_features, new_dim=self.new_dim, func=func
        )
        self.pipeline = None

    def fit(self, X, y):
        pipeline_steps = [('normalizer', Normalizer())]

        if self.use_PCA:
            pipeline_steps.append(('pca', PCA(n_components=self.new_dim)))
        
        pipeline_steps.extend([
            ('rff', self.feature_creator),
            ('classifier', self.classifier)
        ])

        self.pipeline = Pipeline(pipeline_steps).fit(X, y)

        return self

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        return self.pipeline.predict(X)
