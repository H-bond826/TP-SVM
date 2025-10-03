#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from svm_source import *
from sklearn import svm
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from time import time

scaler = StandardScaler()

import warnings
warnings.filterwarnings("ignore")

plt.style.use('ggplot')
# %%
###################################################
#               Iris Dataset
###################################################
iris = datasets.load_iris()
X = iris.data
X = scaler.fit_transform(X)
y = iris.target
X = X[y != 0, :2]
y = y[y != 0]

# Visualization
plt.show()
plt.close("all")
plt.ion()
plt.figure(1, figsize=(10, 5))
plt.title('iris data set')
plot_2d(X, y)

# split train test (say 25% for the test)
# using train_test_split whithout shuffling (fix the random state = 42) for reproductibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# %%
# Q1 Linear kernel

# fit the model and select the best hyperparameter C
parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 200))}
clf1 = SVC()
clf_linear = GridSearchCV(clf1, parameters, n_jobs=-1)
clf_linear.fit(X_train, y_train)

# compute the score
print('Generalization score for linear kernel: %s, %s' %
      (clf_linear.score(X_train, y_train),
       clf_linear.score(X_test, y_test)))

#%%
# Q2 polynomial kernel
Cs = list(np.logspace(-3, 3, 5))
gammas = 10. ** np.arange(1, 2)
degrees = np.r_[1, 2, 3]

# fit the model and select the best set of hyperparameters
parameters = {'kernel': ['poly'], 'C': Cs, 'gamma': gammas, 'degree': degrees}
clf2 = SVC()
clf_poly = GridSearchCV(clf2, parameters, n_jobs=-1)
clf_poly.fit(X_train, y_train)

print(clf_poly.best_params_)
print('Generalization score for polynomial kernel: %s, %s' %
      (clf_poly.score(X_train, y_train),
       clf_poly.score(X_test, y_test)))

#%%
# display the results using frontiere
def f_linear(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_linear.predict(xx.reshape(1, -1))

def f_poly(xx):
    return clf_poly.predict(xx.reshape(1, -1))

# display the frontiere
plt.ion()
plt.figure(figsize=(15, 5))
plt.subplot(131)
plot_2d(X, y)
plt.title("iris dataset")

plt.subplot(132)
frontiere(f_linear, X, y)
plt.title("linear kernel")

plt.subplot(133)
frontiere(f_poly, X, y)

plt.title("polynomial kernel")
plt.tight_layout()
plt.draw()




#%%
###################################################
#               Face Recognition Task
###################################################
"""
The dataset used in this example is a preprocessed excerpt
of the "Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

  _LFW: http://vis-www.cs.umass.edu/lfw/
"""

####################################################################
# Download the data and unzip; then load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4,
                              color=True, funneled=False, slice_=None,
                              download_if_missing=True)
# data_home='.'

# introspect the images arrays to find the shapes (for plotting)
images = lfw_people.images
n_samples, h, w, n_colors = images.shape

# the label to predict is the id of the person
target_names = lfw_people.target_names.tolist()

####################################################################
# Pick a pair to classify such as
names = ['Tony Blair', 'Colin Powell']
# names = ['Donald Rumsfeld', 'Colin Powell']

idx0 = (lfw_people.target == target_names.index(names[0]))
idx1 = (lfw_people.target == target_names.index(names[1]))
images = np.r_[images[idx0], images[idx1]]
n_samples = images.shape[0]
y = np.r_[np.zeros(np.sum(idx0)), np.ones(np.sum(idx1))].astype(int)

# plot a sample set of the data
plot_gallery(images, np.arange(12))
plt.show()

#%%
####################################################################
# Extract features

# features using only illuminations
X = (np.mean(images, axis=3)).reshape(n_samples, -1)

# # or compute features using colors (3 times more features)
# X = images.copy().reshape(n_samples, -1)

# Scale features
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)

#%%
####################################################################
# Split data into a half training and half test set
# X_train, X_test, y_train, y_test, images_train, images_test = \
#    train_test_split(X, y, images, test_size=0.5, random_state=0)
# X_train, X_test, y_train, y_test = \
#    train_test_split(X, y, test_size=0.5, random_state=0)

indices = np.random.permutation(X.shape[0])
train_idx, test_idx = indices[:X.shape[0] // 2], indices[X.shape[0] // 2:]
X_train, X_test = X[train_idx, :], X[test_idx, :]
y_train, y_test = y[train_idx], y[test_idx]
images_train, images_test = images[
    train_idx, :, :, :], images[test_idx, :, :, :]

####################################################################
# Quantitative evaluation of the model quality on the test set

# %%
# Q4
print("--- Linear kernel ---")
print("Fitting the classifier to the training set")
t0 = time()

# fit a classifier (linear) and test all the Cs
Cs = 10. ** np.arange(-5, 6)
scores = []
for C in Cs:
    clf_tmp = SVC(kernel='linear', C=C)
    clf_tmp.fit(X_train, y_train)
    # 这里用测试集分数来挑 C（课程里通常会再套一层验证集/交叉验证；本 TP 用固定半数当测试集）
    scores.append(clf_tmp.score(X_test, y_test))

ind = int(np.argmax(scores))
best_C = Cs[ind]
best_acc = float(scores[ind])
best_err = 1.0 - best_acc

print("Best C: {}".format(best_C))

plt.figure()
plt.plot(Cs, scores, label="Accuracy")
plt.scatter([best_C], [best_acc], s=80, zorder=3)
plt.axvline(best_C, linestyle="--", alpha=0.6)
plt.annotate(
    "Best C={:.1e}\nacc={:.3f}".format(best_C, best_acc),
    xy=(best_C, best_acc),
    xytext=(1.5*best_C, min(1.0, best_acc + 0.05)),  
    arrowprops=dict(arrowstyle="->", lw=1),
    fontsize=10
)
plt.xlabel("Paramètres de régularisation C")
plt.ylabel("Scores d'apprentissage (accuracy)")
plt.xscale("log")
plt.legend()
plt.tight_layout()
plt.show()

print("Best score (accuracy): {}".format(best_acc))

# 或者用Erreur de prédiction，效果是一样的
errors = 1.0 - np.array(scores)

plt.figure()
plt.plot(Cs, errors, label="Erreur de prédiction")
plt.scatter([best_C], [best_err], s=80, zorder=3)
plt.axvline(best_C, linestyle="--", alpha=0.6)
plt.annotate(
    "Best C={:.1e}\nerreur={:.3f}".format(best_C, best_err),
    xy=(best_C, best_err),
    xytext=(1.5*best_C, min(1.0, best_err + 0.05)),
    arrowprops=dict(arrowstyle="->", lw=1),
    fontsize=10
)
plt.xlabel("Paramètre de régularisation C")
plt.ylabel("Erreur de prédiction (1 - accuracy)")
plt.xscale("log")
plt.legend()
plt.tight_layout()
plt.show()


print("Predicting the people names on the testing set")
t0 = time()

#%%
# predict labels for the X_test images with the best classifier
clf = SVC(kernel='linear', C=Cs[ind])
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("done in %0.3fs" % (time() - t0))
# The chance level is the accuracy that will be reached when constantly predicting the majority class.
print("Chance level : %s" % max(np.mean(y), 1. - np.mean(y)))
print("Accuracy : %s" % clf.score(X_test, y_test))

#%%
####################################################################
# Qualitative evaluation of the predictions using matplotlib

prediction_titles = [title(y_pred[i], y_test[i], names)
                     for i in range(y_pred.shape[0])]

plot_gallery(images_test, prediction_titles)
plt.show()

####################################################################
# Look at the coefficients
plt.figure()
plt.imshow(np.reshape(clf.coef_, (h, w)))
plt.show()


# %%
# Q5

def run_svm_cv(_X, _y):
    _indices = np.random.permutation(_X.shape[0])
    _train_idx, _test_idx = _indices[:_X.shape[0] // 2], _indices[_X.shape[0] // 2:]
    _X_train, _X_test = _X[_train_idx, :], _X[_test_idx, :]
    _y_train, _y_test = _y[_train_idx], _y[_test_idx]

    _parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 5))}
    _svr = svm.SVC()
    _clf_linear = GridSearchCV(_svr, _parameters)
    _clf_linear.fit(_X_train, _y_train)

    print('Generalization score for linear kernel: %s, %s \n' %
          (_clf_linear.score(_X_train, _y_train), _clf_linear.score(_X_test, _y_test)))

print("Score sans variable de nuisance")
# use run_svm_cv on original data
run_svm_cv(X, y)

print("Score avec variable de nuisance")
n_features = X.shape[1]
# On rajoute des variables de nuisances
sigma = 1
noise = sigma * np.random.randn(n_samples, 300, ) 
#with gaussian coefficients of std sigma
X_noisy = np.concatenate((X, noise), axis=1)
X_noisy = X_noisy[np.random.permutation(X.shape[0])]
# use run_svm_cv on noisy data
run_svm_cv(X_noisy, y)


#%%
# Q6
print("Score apres reduction de dimension")

# 为了比较不同降维维数的效果，这里测试一组 n_components
# 注意上限不能超过 min(n_samples, n_features_noisy)
max_nc = min(X_noisy.shape[0] // 2, X_noisy.shape[1])  # 留点余量（训练集大小的阶）
grid_n_components = [20, 60, 100]
grid_n_components = [k for k in grid_n_components if k <= max_nc]
if len(grid_n_components) == 0:
    grid_n_components = [min(20, max_nc)]

C_grid = list(np.logspace(-3, 3, 5))

test_scores = []
train_scores = []
best_records = []  # (k, best_C, train_score, test_score)

# 取与前面一致的划分
Xn_train = X_noisy[train_idx, :]
Xn_test  = X_noisy[test_idx, :]
yn_train = y[train_idx]
yn_test  = y[test_idx]

for k in grid_n_components:
    # 1) 在训练集上拟合 PCA（随机 SVD），再同时变换训练/测试
    pca = PCA(n_components=k, svd_solver='randomized')
    Xtr = pca.fit_transform(Xn_train)
    Xte = pca.transform(Xn_test)

    # 2) 线性核 + 扫 C（小网格）做网格搜索
    clf = GridSearchCV(SVC(kernel='linear'), {'C': C_grid}, n_jobs=-1)
    clf.fit(Xtr, yn_train)

    tr = clf.score(Xtr, yn_train)
    te = clf.score(Xte, yn_test)
    train_scores.append(tr)
    test_scores.append(te)
    best_records.append((k, clf.best_params_['C'], tr, te))

# 打印最优的 n_components 及对应 C
best_idx = int(np.argmax(test_scores))
best_k, best_C, best_tr, best_te = best_records[best_idx]
print(f"Best n_components = {best_k}, best C = {best_C}")
print(f"Train score = {best_tr:.3f}, Test score = {best_te:.3f}")

# 画出 测试分数 vs n_components
plt.figure()
plt.plot(grid_n_components, test_scores, marker='o')
plt.xlabel("n_components (PCA)")
plt.ylabel("Test accuracy (linear SVM)")
plt.title("Impact de la dimension apres PCA (donnees bruitees)")
plt.tight_layout()
plt.show()


# %%
