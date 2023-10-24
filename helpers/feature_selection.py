
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

import pandas as pd
import numpy as np
import cmath

class FeatureSelection:
    def __init__(self) -> None:
        print(f"Init class FS {self.__dict__}")

    def calc_n_features(self, n_samples):
        '''
            Thumb rule = N >= D(D-1)/2
        '''
        a = 1
        b = -1
        c = -2 * n_samples

        # calculate the discriminant
        d = (b**2) - (4*a*c)

        # find two solutions
        sol1 = (-b-cmath.sqrt(d))/(2*a)
        sol2 = (-b+cmath.sqrt(d))/(2*a)

        n_features = int(np.max([np.round(np.abs(sol1)), np.round(np.abs(sol2))]))

        return n_features


    def select_bestk_features(self, data, target):

        # Get the best value of k from grid search
        best_k = self.calc_n_features(data.shape[0])

        # Apply Maximum Entropy-based feature selection with the best k
        selector = SelectKBest(k=best_k) # score_func=mutual_info_classif
        X_new = selector.fit_transform(data, target)

        # Get the indices of the selected features
        selected_features = np.where(selector.get_support())[0]

        # Convert X_new to a DataFrame with selected features
        selected_data = pd.DataFrame(X_new, columns=selected_features)

        return selected_data, selected_features
    
    def select_rfe_features(self, data, target):
        estimator = SVC(kernel='linear', decision_function_shape='ovr',class_weight='balanced')

        rfe = RFE(estimator, n_features_to_select=self.calc_n_features(data.shape[0]), verbose=2)
        rfe.fit(data, target)
        
        # Get the indices of the selected features
        selected_features = np.where(rfe.support_)[0]

        # Convert the data to a DataFrame with the selected features
        selected_data = pd.DataFrame(data, columns=selected_features)
        
        # selected_features = rfe.support_
        # selected_data = data[:, selected_features]

        return selected_data, selected_features

