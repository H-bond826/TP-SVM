import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ============ 1) Data: highly imbalanced binary classification in 2D ============
def make_imbalanced_2d(
    n_samples=600,
    weights=(0.9, 0.1),   # 90% vs 10%
    class_sep=1.2,        # inter-class separability (larger -> easier)
    flip_y=0.02,          # small label noise ratio
    random_state=42
):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,         # 2D so we can visualize
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        weights=list(weights),# set class imbalance
        class_sep=class_sep,  # control difficulty
        flip_y=flip_y,        # add a bit of label noise
        random_state=random_state,
    )
    return X, y

# ============ 2) Plot helpers: draw decision boundary like svm_gui.py ============
def plot_decision_function(ax, clf, X, y, fill=True, padding=0.5, grid_res=300):
    """Draw decision_function contours: 0 (decision boundary) and ±1 (margins)."""
    x_min, x_max = X[:,0].min() - padding, X[:,0].max() + padding
    y_min, y_max = X[:,1].min() - padding, X[:,1].max() + padding

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_res),
        np.linspace(y_min, y_max, grid_res)
    )
    # clf is a pipeline (scaler + SVC); feeding raw grid points is fine
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    if fill:
        # background filled contours (optional)
        cs = ax.contourf(xx, yy, Z, levels=20, alpha=0.25)
    # draw the decision boundary (level 0) and the margins (levels ±1)
    ax.contour(xx, yy, Z, levels=[0], linewidths=2)        # decision boundary
    ax.contour(xx, yy, Z, levels=[-1, 1], linestyles="--") # margins

def scatter_data(ax, X, y, sv=None):
    """Scatter points of both classes and optionally highlight support vectors."""
    ax.scatter(X[y==1,0], X[y==1,1], s=25, c='tab:blue', edgecolor='k', label='Class +1')
    ax.scatter(X[y==0,0], X[y==0,1], s=25, edgecolor='k', label='Class 0', c='tab:gray')
    if sv is not None:
        ax.scatter(sv[:,0], sv[:,1], s=90, facecolors='none', edgecolors='k', linewidths=1.5, label='Support Vectors')

# ============ 3) Main: compare multiple C values for linear SVM ============
def main():
    # ---- data and split ----
    X, y = make_imbalanced_2d(n_samples=600, weights=(0.9, 0.1), class_sep=1.0, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

    # C values from large (hard margin tendency) to small (soft margin)
    Cs = [10.0, 1.0, 0.3, 0.1, 0.03, 0.01]

    ncols = 3
    nrows = int(np.ceil(len(Cs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4.2*nrows), squeeze=False)

    print("Class balance (train): {:.1f}% vs {:.1f}%".format(100*(ytr==0).mean(), 100*(ytr==1).mean()))
    print("Class balance (test) : {:.1f}% vs {:.1f}%".format(100*(yte==0).mean(), 100*(yte==1).mean()))
    print("------------------------------------------------------")

    for idx, C in enumerate(Cs):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]

        # pipeline ensures consistent preprocessing (standardization) and modeling
        clf = make_pipeline(
            StandardScaler(),
            SVC(kernel='linear', C=C)
        )
        clf.fit(Xtr, ytr)

        # extract SVC and support vectors for visualization
        svc = clf.named_steps['svc']
        sv_idx = svc.support_    # indices in the training set
        sv_raw = Xtr[sv_idx]     # original coordinates for plotting

        # evaluate
        ypred_tr = clf.predict(Xtr)
        ypred_te = clf.predict(Xte)
        acc_tr = accuracy_score(ytr, ypred_tr)
        acc_te = accuracy_score(yte, ypred_te)
        print(f"C={C:<5}: train acc={acc_tr:.3f} | test acc={acc_te:.3f} | #SV={len(sv_idx)}")

        # plot decision function and data with support vectors
        plot_decision_function(ax, clf, Xtr, ytr, fill=True, padding=0.6)
        scatter_data(ax, Xtr, ytr, sv=sv_raw)

        ax.set_title(f"Linear SVM, C={C}\ntrain={acc_tr:.3f}, test={acc_te:.3f}, SV={len(sv_idx)}")
        ax.set_xlabel("x1"); ax.set_ylabel("x2")
        ax.legend(loc='upper right', fontsize=8)

    # hide empty subplots if any
    for j in range(idx+1, nrows*ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
