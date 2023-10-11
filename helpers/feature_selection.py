
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import GridSearchCV

import pandas as pd
import numpy as np

class FeatureSelection:
    def __init__(self) -> None:
        pass

    def max_entropy_feature_selection(self, data, target, param_grid = {'k': [40, 50, 60]}):
        # Create a SelectKBest instance with mutual information as the scoring function
        selector = SelectKBest(score_func=mutual_info_classif)

        # Perform grid search to find the best k
        grid_search = GridSearchCV(selector, param_grid, cv=5, scoring='accuracy', verbose=2)
        grid_search.fit(data, target)

        # Get the best value of k from grid search
        best_k = grid_search.best_params_['k']

        # Apply Maximum Entropy-based feature selection with the best k
        selector = SelectKBest(score_func=mutual_info_classif, k=best_k)
        X_new = selector.fit_transform(data, target)

        # Get the indices of the selected features
        selected_features = np.where(selector.get_support())[0]

        # Convert X_new to a DataFrame with selected features
        selected_data = pd.DataFrame(X_new, columns=selected_features)

        return selected_data, selected_features
