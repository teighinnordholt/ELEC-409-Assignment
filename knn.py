#external imports
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from scipy.stats import fisher_exact
from scipy.stats import chi2_contingency


#internal imports
from data_reader import create_datasets

fpath = 'Instructions/Dataset_C_MD_outcome2.xlsx'

#first index: patient
#second index: outcome first, then gene expressions
training, testing = create_datasets(fpath, rand=True) #false for repeatibility, true for genuine dataset

#seperating training set into gene expression array (x) to predict outcome (y)
x_train = training[:,1:]
y_train = training[:, 0]

#seperating testing set into gene expression array (x) to validate outcome (y)
x_test = testing[:,1:]
y_test = testing[:, 0]

#setting up ks to test
max_k = 30
k_range = range(1, max_k+1)

#defining leave one out validator
loo = LeaveOneOut()

#array for accuracies of each k
accuracies = np.zeros(len(k_range))

for train_index, test_index in loo.split(x_train):
    X_train, X_test = x_train[train_index], x_train[test_index]
    Y_train, Y_test = y_train[train_index], y_train[test_index]

    #iterate through each k value, train, and predict
    for i, k in enumerate(k_range):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, Y_train)
        y_pred = knn.predict(X_test)

        accuracies[i] += accuracy_score(Y_test, y_pred)

#finding k with largest score
best_k = np.argmax(accuracies) + 1
print(accuracies)

print(f'Best k for model is {best_k} with a score of {accuracies[best_k]} out of {max_k} in loo testing')

#training best classifier
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(x_train, y_train)

#predicting on test data
y_pred = knn_best.predict(x_test)

#evaluating performance
TP = np.sum((y_test == 1) & (y_pred == 1))
TN = np.sum((y_test == 0) & (y_pred == 0))
FP = np.sum((y_test == 0) & (y_pred == 1))
FN = np.sum((y_test == 1) & (y_pred == 0))

print(f'True Positives (TP): {TP}')
print(f'True Negatives (TN): {TN}')
print(f'False Positives (FP): {FP}')
print(f'False Negatives (FN): {FN}')

contingency_table = np.array([[TP, FP],
                               [FN, TN]])

#calculate Fisher's Exact Test
odds_ratio, p_value = fisher_exact(contingency_table)
print(f'Fisher\'s Exact Test: Odds Ratio = {odds_ratio}, p-value = {p_value}')

#calculate Matthews' Correlation Coefficient
phi = matthews_corrcoef(y_test, y_pred)
print(f'Matthews\' Correlation Coefficient: {phi}')