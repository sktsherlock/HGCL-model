

from statistics import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import  SVC
#from thundersvm import SVC
from sklearn.model_selection import  StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np
# import os
# import torch
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# if torch.cuda.is_available():
#     torch.device = "cuda:1"
# else:
#     torch.device = "cpu"

# 
# Create synthetic dataset of 100000 samples
# X, y = make_classification(n_samples=100000, n_features=20, n_informative=17, n_redundant=3, random_state=5)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=5)



# # Fit the model to training data
# model.fit(X_train, y_train)
# Check test set accuracy
# accuracy = model.score(X_test, y_test)
# print('Accuracy: {}'.format(accuracy))
# # %%
X = np.load("svm_embedding.npz")
Y = np.load("svm_label.npz")
for i in range(4):
    # X_train, X_test, y_train, y_test = train_test_split(X['arr_{}'.format(i)], Y['arr_{}'.format(i)], test_size=.2, random_state=5)
    x = X['arr_{}'.format(i)]
    y = Y['arr_{}'.format(i)]
    print(x.shape)
    params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = SVC(C=1, kernel='rbf')
        model.fit(x_train, y_train)
        #scores = cross_val_score(model, x_train, y_train, cv=5)
        accuracy = model.score(x_test, y_test)
        print('Accuracy: {}'.format(accuracy))
        #classifier = GridSearchCV(SVC( ), params, cv=5, scoring='accuracy', verbose=0)
        #classifier.fit(x_train, y_train)
        #accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    #print(np.mean(accuracies), np.std(accuracies))
        #return np.mean(accuracies), np.std(accuracies)

        # model = SVC(C=C, kernel='rbf')
        # model.fit(X_train, y_train)
        # scores = cross_val_score(model, iris.data, iris.target, cv=5)
        # accuracy = model.score(X_test, y_test)
        # print('Accuracy: {}'.format(accuracy))






# def svc(embeds, labels):
#     params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
#     kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
#     accuracies = []
#     for train_index, test_index in kf.split(x, y):
#         x_train, x_test = x[train_index], x[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
#         classifier.fit(x_train, y_train)
#         accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
#     return np.mean(accuracies), np.std(accuracies)

# #In[]
# # import necessary libraries
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# #In[]
# # make a timer class to measure speedup
# import time

# class Timer:    
#     def __enter__(self):
#         self.tick = time.time()
#         return self

#     def __exit__(self, *args, **kwargs):
#         self.tock = time.time()
#         self.elapsed = self.tock - self.tick
# #In[]
# # Create synthetic dataset of 100000 samples
# X, y = make_classification(n_samples=100000, n_features=20, n_informative=17, n_redundant=3, random_state=5)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=5)

# #In[]
# # For normal support vector classifier
# from sklearn.svm import SVC

# # initializing classifier
# clf = SVC(C=100)

# # fitting to training data
# with Timer() as clf_fit:
#     clf.fit(X_train, y_train)

# # predicting test labels
# with Timer() as clf_predict:
#     clf.predict(X_test)

# # calculating test set accuracy
# with Timer() as clf_score:
#     clf.score(X_test, y_test)

# print('time elapsed(seconds) for Fitting: {}, Prediction: {}, Scoring: {}'.format(clf_fit.elapsed, clf_predict.elapsed, clf_score.elapsed))
# #In[]
# # For ThunderSVM support vector classifier
# from thundersvm import SVC

# model = SVC(C=100)
# with Timer() as model_fit:
#     model.fit(X_train, y_train)

# with Timer() as model_predict:
#     model.predict(X_test)

# with Timer() as model_score:
#     model.score(X_test, y_test)

# print('time elapsed(seconds) for Fitting: {}, Prediction: {}, Scoring: {}'.format(model_fit.elapsed, model_predict.elapsed, model_score.elapsed))
# # %%
# print('ThunderSVM Fit Speedup with 100,000 samples: {:.1f}x'.format(clf_fit.elapsed/model_fit.elapsed))
# print('ThunderSVM Predict Speedup with 100,000 samples: {:.1f}x'.format(clf_predict.elapsed/model_predict.elapsed))
# print('ThunderSVM Score Speedup with 100,000 samples: {:.1f}x'.format(clf_score.elapsed/model_score.elapsed))
# # %%
