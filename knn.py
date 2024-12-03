#external imports
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from scipy.stats import fisher_exact
from scipy.stats import chi2_contingency

#internal imports
from data_reader import create_datasets

fpath = 'Instructions/Dataset_C_MD_outcome2.xlsx'

#setting up ks to test
max_k = 25
k_range = range(1, max_k+1)

#number of random dataset splits to test
num_its = 2

#parameters to 
fishers = np.zeros(num_its)
matthews = np.zeros(num_its)
best_ks = np.zeros(num_its)

for it in range(num_its):
    print(f'\nIteration: {it+1}')
    #first index: patient
    #second index: outcome first, then gene expressions
    training, testing = create_datasets(fpath, rand=True) #false for repeatibility, true for genuine random split

    #seperating training set into gene expression array (x) to predict outcome (y)
    x_train = training[:,1:]
    y_train = training[:, 0]

    #seperating testing set into gene expression array (x) to validate outcome (y)
    x_test = testing[:,1:]
    y_test = testing[:, 0]

    #defining leave one out validator
    loo = LeaveOneOut()

    #array for accuracies of each k
    accuracies = np.zeros(len(k_range))

    for train_index, test_index in loo.split(x_train):
        X_train, X_test = x_train[train_index], x_train[test_index]
        Y_train, Y_test = y_train[train_index], y_train[test_index]

        #iterate through each k value, train, and predict
        for i, k in enumerate(k_range):
            knn = KNeighborsClassifier(n_neighbors=k, algorithm='ball_tree', metric='euclidean', weights='distance') #more efficient algorithm but keep euclidean distance
            knn.fit(X_train, Y_train)
            y_pred = knn.predict(X_test)

            accuracies[i] += accuracy_score(Y_test, y_pred)

    #finding k with largest score
    best_k = np.argmax(accuracies) + 1
    best_ks[it] = best_k
    #print(f'Best k for model is {best_k} with a score of {accuracies[best_k-1]} out of {max_k} in loo testing')

    #training best classifier
    knn_best = KNeighborsClassifier(n_neighbors=best_k, algorithm='ball_tree', metric='euclidean', weights='distance')
    knn_best.fit(x_train, y_train)

    #predicting on test data
    y_pred = knn_best.predict(x_test)

    #print(f'Test: {y_test}')
    #print(f'Pred: {y_pred}')

    #evaluating performance
    TP = np.sum((y_test == 1) & (y_pred == 1))
    TN = np.sum((y_test == 0) & (y_pred == 0))
    FP = np.sum((y_test == 0) & (y_pred == 1))
    FN = np.sum((y_test == 1) & (y_pred == 0))

    if 0:
        print(f'True Positives (TP): {TP}')
        print(f'True Negatives (TN): {TN}')
        print(f'False Positives (FP): {FP}')
        print(f'False Negatives (FN): {FN}')

    contingency_table = np.array([[TP, FP],
                                [FN, TN]])
    
    #calculate Fisher's Exact Test
    odds_ratio, p_value = fisher_exact(contingency_table)
    fishers[it] = p_value

    #calculate Matthews' Correlation Coefficient
    phi = matthews_corrcoef(y_test, y_pred)
    matthews[it] = phi

    if 1:
        print(f'Fisher\'s Exact Test: {p_value}')
        print(f'Matthews\' Correlation Coefficient: {phi}')


print(f'\nFisher\'s Exact Test: {np.mean(fishers)} ± {np.std(fishers)}')
print(f'Matthew\'s Correlation Coefficient: {np.mean(matthews)} ± {np.std(matthews)}')

# plotting fishers

plt.scatter(range(num_its), fishers, color='blue')

plt.axhline(np.mean(fishers), color='blue', label=f'Fisher\'s Exact Test: {np.mean(fishers):.2f} ± {np.std(fishers):.2f}')
plt.axhline(np.mean(fishers) + np.std(fishers), linestyle='--', color='blue')
plt.axhline(np.mean(fishers) - np.std(fishers), linestyle='--', color='blue')

plt.xticks([])
plt.ylim([0,1])

plt.legend(loc='best')
plt.savefig('Outputs/fishers_plot.png', dpi=800)
plt.close()

#plotting mcc

plt.scatter(range(num_its), matthews, color='orange')

plt.axhline(np.mean(matthews), color='orange', label=f'Matthews\' Correlation Coefficient: {np.mean(matthews):.2f} ± {np.std(matthews):.2f}')
plt.axhline(np.mean(matthews) + np.std(matthews), linestyle='--', color='orange')
plt.axhline(np.mean(matthews) - np.std(matthews), linestyle='--', color='orange')

plt.xticks([])
plt.ylim([-1,1])

plt.legend(loc='best')
plt.savefig('Outputs/mcc_plot.png', dpi=800)
plt.close()

#plotting fishers and mcc

plt.scatter(range(num_its), fishers, color='blue')

plt.axhline(np.mean(fishers), color='blue', label=f'Fisher\'s Exact Test: {np.mean(fishers):.2f} ± {np.std(fishers):.2f}')
plt.axhline(np.mean(fishers) + np.std(fishers), linestyle='--', color='blue')
plt.axhline(np.mean(fishers) - np.std(fishers), linestyle='--', color='blue')

plt.scatter(range(num_its), matthews, color='orange')

plt.axhline(np.mean(matthews), color='orange', label=f'Matthews\' Correlation Coefficient: {np.mean(matthews):.2f} ± {np.std(matthews):.2f}')
plt.axhline(np.mean(matthews) + np.std(matthews), linestyle='--', color='orange')
plt.axhline(np.mean(matthews) - np.std(matthews), linestyle='--', color='orange')

plt.xticks([])
plt.ylim([-1,1])

plt.legend(loc='best')
plt.savefig('Outputs/significance_plots.png', dpi=800)
#plt.show()
plt.close()

#plotting ks

plt.scatter(range(num_its), best_ks, color='blue')
plt.axhline(np.mean(best_ks), color='blue', label=f'Mean optimal k value: {np.mean(best_ks):.2f}')

plt.xticks([])

plt.legend(loc='best')
plt.savefig('Outputs/optimal_k.png', dpi=800)