
# src/clustered_cart.py
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

class ClusteredCART(BaseEstimator, ClassifierMixin):

    """
    Mixture duro: K clusters con KMeans + 1 CART por clúster.
    Compatible con GridSearchCV (expone hiperparámetros del árbol).
    """
    def __init__(
        self,
        n_clusters=3,
        random_state=42,
        # Hiperparámetros del árbol (expuestos para GridSearch):
        criterion="gini",
        max_depth=5,
        min_samples_leaf=5,
        min_samples_split=10,
        ccp_alpha=0.0,
        class_weight=None
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.ccp_alpha = ccp_alpha
        self.class_weight = class_weight

    def _new_tree(self):
        return DecisionTreeClassifier(
            random_state=self.random_state,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
            ccp_alpha=self.ccp_alpha,
            class_weight=self.class_weight
        )

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        # 1) KMeans (sobre features ya preprocesadas por el Pipeline)
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters, n_init="auto", random_state=self.random_state
        )
        labels = self.kmeans_.fit_predict(X)

        # Guardamos clases globales (para ordenar predict_proba consistentemente)
        self.classes_ = np.unique(y)

        # 2) Entrenar 1 árbol por clúster
        self.trees_ = []
        for c in range(self.n_clusters):
            idx = np.where(labels == c)[0]
            clf = self._new_tree()
            if len(idx) == 0:
                # Si un clúster queda vacío en un fold, entrenar un árbol "fallback" global
                clf.fit(X, y)
            else:
                clf.fit(X[idx], y[idx])
            self.trees_.append(clf)
        return self

    def _proba_for_tree(self, clf, X):
        """Devolver proba con dos columnas en el orden de self.classes_.
        Maneja el caso donde el árbol haya visto una sola clase en su entrenamiento."""
        if hasattr(clf, "predict_proba"):
            # Probabilidades en el orden clf.classes_
            p = clf.predict_proba(X)
            # Mapear al orden de self.classes_
            out = np.zeros((X.shape[0], len(self.classes_)))
            for j, cls in enumerate(clf.classes_):
                col = np.where(self.classes_ == cls)[0][0]
                out[:, col] = p[:, j]
            return out
        # Sin predict_proba: usar predicción dura y convertir a probas 0/1
        yhat = clf.predict(X)
        out = np.zeros((X.shape[0], len(self.classes_)))
        for i, cls in enumerate(self.classes_):
            out[:, i] = (yhat == cls).astype(float)
        return out

    def predict_proba(self, X):
        check_is_fitted(self, "kmeans_")
        X = np.asarray(X)
        labels = self.kmeans_.predict(X)
        proba = np.zeros((X.shape[0], len(self.classes_)))
        for c, clf in enumerate(self.trees_):
            idx = np.where(labels == c)[0]
            if idx.size > 0:
                proba[idx] = self._proba_for_tree(clf, X[idx])
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        # Elegimos la clase con mayor probabilidad
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]
