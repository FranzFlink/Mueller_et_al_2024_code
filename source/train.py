import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import os
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from scipy.ndimage import gaussian_filter
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict, GridSearchCV, RandomizedSearchCV
from helper import calc_local_roughness

def train_random_forest(training_ds, n_jobs=48):


    training_ds = calc_local_roughness(test_ds_v4)

    X, y = test_ds_v4['BT_2D'].values, test_ds_v4['label'].values
    X_add = test_ds_v4['sur_rgh'].values, test_ds_v4['neighbor_mean'].values, test_ds_v4['neighbor_std'].values


    base_path = f"/projekt_agmwend/data/HALO-AC3/05_VELOX_Tools/ml_sea-ice-classification/models/rf"
    model_path = os.path.join('/projekt_agmwend/data/HALO-AC3/05_VELOX_Tools/ml_sea-ice-classification/models', 'random_forest_model_all_flights_5_bands.joblib')

    # # Check if model exists
    # if os.path.exists(model_path):
    #     print(f"Loading model from {model_path}...")
    #     clf = joblib.load(model_path)
    # else:

    groups = test_ds_v4.time.dt.dayofyear.values
    unique_groups = np.unique(groups)
    print(unique_groups)




    search_params_rf = {
        'n_estimators': [100],
        'max_depth': [50],
        'min_samples_split': [2],
        'min_samples_leaf': [2],
        'bootstrap': [True]
    }


    # Reshape the data to 2D
    X = X.reshape(-1, 5)
    y = y.reshape(-1)

    print('Inital unique labels:', np.unique(y))
    
    print('X shape:', X.shape)
    print('y shape:', y.shape)

    if X_rgh is not None:
        r1, r2, r3 = X_rgh
        r1 = r1.flatten()
        r2 = r2.flatten()
        r3 = r3.flatten()

    # Obscure masking of input pixels as label 3 is inconsistent 
    mask = (y !=0 ) & ~np.isnan(y) & np.all(np.isfinite(X), axis=1) & np.isfinite(r1) & (y!=3)

    # Remove pixels with label 0
    X_filtered = X[mask, :] 
    y_filtered = y[mask]
    r1 = r1[mask]
    r2 = r2[mask]
    r3 = r3[mask]

    dx1 = X_filtered[:, 1] - X_filtered[:, 4]
    dx2 = X_filtered[:, 2] - X_filtered[:, 4]
    dx3 = X_filtered[:, 3] - X_filtered[:, 4]



    X_filtered = np.stack((X_filtered[:, 0], dx1, dx2, dx3, r1, r2, r3), axis=1)

    # Tweaking the labels

    print('Unique labels:', np.unique(y_filtered))

    N = len(y_filtered)

    print(len(y_filtered[y_filtered==1]) / N, 'Open Water')
    print(len(y_filtered[y_filtered==2]) / N, 'Dark Nilas')
    print(len(y_filtered[y_filtered==3]) / N, 'Bare Sea Ice')
    print(len(y_filtered[y_filtered==4]) / N, 'Snow-covered Sea Ice')


    y_filtered[y_filtered==1] = 1
    y_filtered[y_filtered==2] = 2
    y_filtered[y_filtered==3] = 3
    y_filtered[y_filtered==4] = 4

    groups = test_ds_v4.time.dt.dayofyear.values
    pixel_groups = np.outer(groups, np.ones((test_ds_v4.x.size, test_ds_v4.y.size))).flatten()
    pixel_groups = pixel_groups[mask]

    print('Unique labels:', np.unique(y_filtered))
    print('X shape:', X_filtered.shape)
    print('y shape:', y_filtered.shape)

    # Train the Random Forest classifier
    model = RandomForestClassifier(random_state=42, n_jobs=n_jobs,
            n_estimators=50, max_depth=50, min_samples_split=2, min_samples_leaf=1, bootstrap=True,
            criterion='gini', max_features='sqrt',)
    logo = LeaveOneGroupOut()

    # Predict the labels for the entire dataset

    scoring = 'accuracy'

    model.fit(X_filtered, y_filtered)

    #model = search_cv.fit(X_filtered, y_filtered, groups=pixel_groups).best_estimator_

    #model = search_cv.fit(X_filtered, y_filtered, groups=pixel_groups)
    

    y_pred = cross_val_predict(model, X_filtered, y_filtered, cv=logo, n_jobs=1, verbose=5, groups=pixel_groups)# else:

    # clf, y_pred, metrics = train_random_forest(X, y, n_jobs=64)

    #y_pred = model.predict(X_filtered)

    
    # Reshape predictions back to original shape
    # y_pred = y_pred.reshape(y.shape)
    
    # Apply Gaussian smoothing to the predictions
    #y_pred_smoothed = gaussian_filter(y_pred, sigma=sigma)

    #score = clf.score(X_test, y_test)

    # Calculate performance metrics
    accuracy = accuracy_score(y_filtered, y_pred)
    class_report = classification_report(y_filtered, y_pred, output_dict=True)
    conf_mat = confusion_matrix(y_filtered, y_pred)
    
    print('Accuracy:', accuracy)
    print('Classification Report:', class_report)

    metrics = {
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_mat
    }
    
    return model, y_pred, metrics



clf, y_pred, metrics = train_random_forest(X, y, n_jobs=64, X_rgh=X_add)
