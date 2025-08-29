
# src/train.py
import argparse
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

from clustered_cart import ClusteredCART
from data_pipeline import load_titanic, make_preprocessor, train_test_split_strat
from plots import plot_roc_holdout, plot_confusion, plot_clusters_pca, plot_dendrogram_sample

def main():
    parser = argparse.ArgumentParser(description="K-Means + CART (mixture duro) para Titanic")
    parser.add_argument("--cv", type=int, default=5, help="N° folds de cross-validation")
    parser.add_argument("--scoring", type=str, default="f1", help="Métrica de GridSearchCV (default: f1)")
    args = parser.parse_args()

    # 1) Datos
    X, y, num_cols, cat_cols = load_titanic()
    Xtr, Xte, ytr, yte = train_test_split_strat(X, y)

    # 2) Preprocesador (sin fuga)
    pre = make_preprocessor(num_cols, cat_cols)

    # 3) Pipeline completo: pre → ClusteredCART
    pipe = Pipeline(steps=[
        ("pre", pre),
        ("clf", ClusteredCART()),
    ])

    # 4) Grid de hiperparámetros (K + árbol)
    param_grid = {
        "clf__n_clusters": [2, 3],
        "clf__criterion": ["gini", "entropy"],
        "clf__max_depth": [5, 7, 9],
        "clf__min_samples_leaf": [2, 3, 4, 5],
        "clf__min_samples_split": [6, 10, 15],
        "clf__ccp_alpha": [0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012],
        "clf__class_weight": ["balanced"]
    }

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=args.cv,
        scoring=args.scoring,
        n_jobs=-1,
        verbose=1,
    )

    # 5) Entrenar GridSearch (CV)
    gs.fit(Xtr, ytr)
    print("\n==> Mejores hiperparámetros:", gs.best_params_)
    print("==> Mejor score (CV, {}): {:.4f}".format(args.scoring, gs.best_score_))

    # 6) Evaluación hold-out
    yhat = gs.predict(Xte)
    # AUC (si hay predict_proba)
    try:
        yprob = gs.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, yprob)
    except Exception:
        auc = np.nan

    print("\n==> HOLD-OUT (20%)")
    print("Accuracy:", accuracy_score(yte, yhat))
    print("F1:", f1_score(yte, yhat))
    print("AUC:", auc)
    print("\nReporte por clase:\n", classification_report(yte, yhat))
    print("Matriz de confusión:\n", confusion_matrix(yte, yhat))
    out_roc = plot_roc_holdout(yte, yprob, outdir="plots")
    out_cm  = plot_confusion(yte, yhat, outdir="plots")
    print(f"\nGuardado: {out_roc}\nGuardado: {out_cm}")

    # === Obtener X_test preprocesado y labels de KMeans ya entrenado ===
    # gs.best_estimator_ es Pipeline(pre -> clf)
    best_pipe = gs.best_estimator_
    pre = best_pipe.named_steps["pre"]
    clf = best_pipe.named_steps["clf"]

    # Transformar Xte con el preprocesador para obtener la matriz numérica final:
    Xte_pre = pre.transform(Xte)        # numpy array
    clusters_test = clf.kmeans_.predict(Xte_pre)  # labels 0..K-1

    # Gráfico 2D (PCA) con clusters y supervivencia real:
    out_pca = plot_clusters_pca(Xte_pre, clusters_test, yte.values, outdir="plots")
    print(f"Guardado: {out_pca}")

    # Dendrograma (sobre muestra de X_test preprocesado):
    out_dend = plot_dendrogram_sample(Xte_pre, sample_size=150, outdir="plots")
    print(f"Guardado: {out_dend}")

if __name__ == "__main__":
    main()
