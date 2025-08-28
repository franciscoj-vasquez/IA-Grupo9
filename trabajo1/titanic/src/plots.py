
# src/plots.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram

def ensure_dir(path="plots"):
    os.makedirs(path, exist_ok=True)
    return path

def plot_roc_holdout(y_true, y_prob, outdir="plots", fname="roc_holdout.png"):
    ensure_dir(outdir)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1],"--", lw=1, color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC – Hold-out")
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(outdir, fname)
    plt.savefig(path, dpi=150)
    plt.close()
    return path

def plot_confusion(y_true, y_pred, outdir="plots", fname="cm_holdout.png"):
    ensure_dir(outdir)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=["No","Sí"], yticklabels=["No","Sí"])
    plt.xlabel("Predicción"); plt.ylabel("Real")
    plt.title("Matriz de confusión – Hold-out")
    plt.tight_layout()
    path = os.path.join(outdir, fname)
    plt.savefig(path, dpi=150)
    plt.close()
    return path

def plot_clusters_pca(X_pre, cluster_labels, y_true, outdir="plots", fname="clusters_pca.png"):
    """
    X_pre: features ya preprocesadas (num escaladas + dummies)
    cluster_labels: labels de KMeans.predict(X_pre)
    y_true: 0/1 Survived (para colorear o marcar)
    """
    ensure_dir(outdir)
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X_pre)
    plt.figure(figsize=(6,5))
    # color por cluster, marker por supervivencia
    for c in np.unique(cluster_labels):
        mask = (cluster_labels == c)
        plt.scatter(Z[mask,0], Z[mask,1],
                    s=25, alpha=0.7, label=f"Cluster {c}")
    # remarcar sobrevivientes con borde
    survived = (y_true == 1)
    plt.scatter(Z[survived,0], Z[survived,1],
                s=40, facecolors="none", edgecolors="black", linewidths=1, label="Sobrevivió")
    plt.title("Clusters K-Means (PCA 2D) – Hold-out")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    path = os.path.join(outdir, fname)
    plt.savefig(path, dpi=150)
    plt.close()
    return path

def plot_dendrogram_sample(X_pre, sample_size=150, outdir="plots", fname="dendrogram_sample.png"):
    """
    Dendrograma con clustering jerárquico (ward) sobre una muestra de X_pre.
    """
    ensure_dir(outdir)
    n = X_pre.shape[0]
    idx = np.random.RandomState(42).choice(n, size=min(sample_size, n), replace=False)
    Z = linkage(X_pre[idx], method="ward")
    plt.figure(figsize=(8,4))
    dendrogram(Z, no_labels=True, count_sort=True, color_threshold=None)
    plt.title(f"Dendrograma (ward) – muestra n={len(idx)}")
    plt.tight_layout()
    path = os.path.join(outdir, fname)
    plt.savefig(path, dpi=150)
    plt.close()
    return path
