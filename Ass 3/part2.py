import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import statistics as st

def warn(*args, **kwarges):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category= DataConversionWarning)
warnings.filterwarnings("ignore", category= FutureWarning)
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn import metrics

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
import csv



Label = "Credit"
Features = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16","A17","A18","A19"]

def saveBestModel(clf):
    pickle.dump(clf, open("bestModel.model", 'wb'))

def readData(file):
    df = pd.read_csv(file)
    return df

def trainOnAllData(df, clf):
    #Use this function for part 4, once you have selected the best model
    print("TODO")

    saveBestModel(clf)

df = readData("credit_train.csv")
# =============================================================================
# print(df)
# print(df.shape[0])
# print(df.shape[1])
# print(df.loc[0][0:19])
# =============================================================================

classifiers = {"Logistic Regression": LogisticRegression(),
               "Naive Bayes": GaussianNB(),
               "SVM": SVC(),
               "Decision Tree": DecisionTreeClassifier(),
               "Random Forest": RandomForestClassifier(),
               "KNN": KNeighborsClassifier(),
               "Gaussian Process Classifier": GaussianProcessClassifier()
               }

group = GroupKFold(n_splits = 10)
X = np.empty(shape = (500, 19))
y = np.empty(shape = (500, 1))
label = []
for i in range(500):
    label.append(i)
    
for i in range(df.shape[0]):
    temp = []
    for j in range(19):
        temp.append(df.loc[i][j])
    X[i] = temp
    if df.loc[i][19] == "good":
        y[i] = [1]
    else:
        y[i] = [0]
    
# =============================================================================
# print(X[0])
# print(y)
# =============================================================================


for name, clf in zip(classifiers.keys(), classifiers.values()):
    auc_list = []
    for train_index, test_index in group.split(X, y, label):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = clf.fit(X_train, y_train)
        pred_y = model.predict(X_test)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_y)
        auc_list.append(metrics.auc(fpr, tpr))
    avg = sum(auc_list)/len(auc_list)
    sd = st.stdev(auc_list)
    print(name, ": ", avg, " ", sd, "\n")


#SVC parameter test
parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.01, 0.1, 1, 0.001, 0.0001]}
grid_search = GridSearchCV(SVC(), parameters, scoring='roc_auc', cv = 10)
grid_search.fit(X, y)
print("Best score: ", grid_search.best_score_)
print("For SVC \nBest C is", grid_search.best_estimator_.C, "\nBest gamma is", grid_search.best_estimator_.gamma)
print()
#Random Forest parameter test
parameters_2 = {'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)], 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
grid_search_2 = GridSearchCV(RandomForestClassifier(), parameters_2, scoring='roc_auc', cv = 10)
grid_search_2.fit(X, y)
print("Best score: ", grid_search_2.best_score_)
print("For Random Forest \nBest max_depth is", grid_search_2.best_estimator_.max_depth, "\nBest n_estimators is", grid_search_2.best_estimator_.n_estimators)



#To get confusion matrix of best model
f = open("bestModel.output", "w")  
writer = csv.writer(f)
writer.writerow(('A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'Credit', 'Prediction'))
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1, shuffle = True)
clf_best = RandomForestClassifier(n_estimators=800, max_depth=20)
model_best = clf_best.fit(X_tr, y_tr)
prediction_y = model_best.predict(X_te)
tn, fp, fn, tp = confusion_matrix(y_te, prediction_y).ravel()
fpr_b, tpr_b, thresholds_b = metrics.roc_curve(y_te, prediction_y)
print(metrics.confusion_matrix(y_te, prediction_y))
print("Precision is ", tp/(tp+fp))
print("Recall is ", tp/(tp+fn))
print("Accuracy is ", (tp+tn)/(tn+fp+fn+tp))
print("AUC is ", metrics.auc(fpr_b, tpr_b))
total_pred_y = model_best.predict(X)
write_credit = []
write_pred_y = []
for i in range(len(y)):
    if y[i] == 1:
        write_credit.append("good")
    else:
        write_credit.append("bad")
        
for i in range(len(total_pred_y)):
    if total_pred_y[i] == 1:
        write_pred_y.append("good")
    else:
        write_pred_y.append("bad")
        
for i in range(len(X)):
    writer.writerow((X[i][0],X[i][1],X[i][2],X[i][3],X[i][4],X[i][5],X[i][6],X[i][7],X[i][8],X[i][9],X[i][10],X[i][11],X[i][12],X[i][13],X[i][14],X[i][15],X[i][16],X[i][17],X[i][18],write_credit[i],write_pred_y[i]))

f.close()  


#Save best model
last_clf = RandomForestClassifier(n_estimators=800, max_depth=20)
last_model = last_clf.fit(X, y)
saveBestModel(last_model)