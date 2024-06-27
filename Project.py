# Import pandas
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_selection import VarianceThreshold
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ****************** Import the dataset ******************
def import_dataset(name_dataset, names_columns = None):
    if names_columns == None:
        data = pd.read_csv(name_dataset)
    else:
        data = pd.read_csv(name_dataset, names = names_columns)
    data = data.applymap(lambda v:(v.strip('\t') if type(v)==str else v))
    return data

# ****************** Pre-processing ******************

# Replace missing values by average or median values
def replace_missing_values(data):
    fillna_mean_cols = pd.Index(set(data.columns[data.dtypes == "float64"]))
    fillna_most_cols = pd.Index(set(data.columns[data.dtypes == "object"]))
    data[fillna_mean_cols] = data[fillna_mean_cols].fillna(data[fillna_mean_cols].mean())
    if len(fillna_most_cols) != 0:
        data[fillna_most_cols] = data[fillna_most_cols].fillna(data[fillna_most_cols].mode().iloc[0])
    y = data["classification"]
    data = data.drop(columns="classification")
    data = pd.get_dummies(data, drop_first=True)
    return data, y

# Center and normalize the data
def center_normalize(data):
    data = ( data - np.resize(np.mean(data,axis=0), np.shape(data)) ) / np.sqrt(np.var(data, axis=0))
    return data

# ****************** Split the dataset
# Split between training set and test set 
def split_training_test(X, y,train_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)
    return(X_train, X_test, y_train, y_test)

# ****************** Train & validate the model (including feature selection)
# We use a variance threshold for feature selection
def feature_selection(var_thrs : float, X) :
    X = X.drop(columns="id", errors="ignore") # Drop the id column, useless for detection
    sel = VarianceThreshold(threshold=var_thrs)
    X_selected = sel.fit_transform(X)
    return X_selected

# Compute the cross-validation of our estimator
def cross_validation(estimator, X_train, y_train) :
    score = cross_val_score(estimator, X_train, y_train, cv=3)
    return score

def train_validate_model(estimator, X_train, y_train, X_test, y_test) :
    # Model training
    estimator.fit(X_train, y_train)
    # Model accuracy based on the training set
    model_accuracy_score = estimator.score(X_train, y_train)
    # Model prediction
    y_pred = estimator.predict(X_test)
    # Model accuracy based on the prediction
    cm = confusion_matrix(y_test, y_pred)
    return y_pred, model_accuracy_score, cm


# ****************** MAIN ******************
def main(name_dataset, names_columns = None, estimator_name = "All", var_thrs = 0.05, train_size = 0.5):
    data = import_dataset(name_dataset, names_columns)
    X, y = replace_missing_values(data)
    X = feature_selection(var_thrs, X)
    X = center_normalize(X)
    X_train, X_test, y_train, y_test = split_training_test(X, y,train_size)
    estimators_dict = {
        "Linear SVC" : LinearSVC(),
        "Naive Bayes" : GaussianNB(),
        "SGD Classifier" : SGDClassifier(),
        "KNeighbors Classifier" : KNeighborsClassifier(),
        "Random Forest" : RandomForestClassifier()
                        }
    if estimator_name != "All" :
        cross_val_score_value = cross_validation(estimators_dict[estimator_name], X_train, y_train)
        y_pred, model_accuracy_score, cm = train_validate_model(estimators_dict[estimator_name], X_train, y_train, X_test, y_test)
        return {estimator_name: (y_pred, model_accuracy_score, cm, cross_val_score_value)}
    else :
        results_dict = {name : [] for name in estimators_dict.keys()}
        for esti_name in estimators_dict.keys() :
            cross_val_score_value = cross_validation(estimators_dict[esti_name], X_train, y_train)
            y_pred, model_accuracy_score, cm = train_validate_model(estimators_dict[esti_name], X_train, y_train, X_test, y_test)
            results_dict[esti_name] = y_pred, model_accuracy_score, cm, cross_val_score_value
        return results_dict


# This part is used to test the main function
if False :
    file_location = r"C:\Users\tranc\Documents\IMT Atlantique\ML\Project\projet-machine-learning\data_banknote_authentication.txt"
    r_ = main(file_location, estimator_name="Linear SVC", names_columns = [ "Variance of Wavelet Transformed image", \
                                                "Skewness of Wavelet Transformed image", \
                                                "Curtosis of Wavelet Transformed image", \
                                                "Entropy of image", "classification"])
    #file_location = r"C:\Users\tranc\Documents\IMT Atlantique\ML\Project\projet-machine-learning\kidney_disease.csv"
    #r_ = main(file_location, estimator_name="Linear SVC")
    for key in r_.keys() :
        print(key)
        print(f"y_pred : \n{set(r_[key][0])}")
        print(f"Model accuracy : {r_[key][1]}")
        print(f"Confusion matrix :\n{r_[key][2]}")
        print(f"Cross val score : {r_[key][3]}")
        print()
