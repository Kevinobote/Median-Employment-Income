import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from surprise import Dataset, Reader, SVD, KNNBasic

def detect_anomalies(df, columns, method='isolation_forest'):
    """
    Detect anomalies in the dataset using specified method
    """
    try:
        X = df[columns].values
        if method == 'isolation_forest':
            clf = IsolationForest(random_state=42)
        elif method == 'one_class_svm':
            clf = OneClassSVM(kernel='rbf')
        
        # Fit and predict
        predictions = clf.fit_predict(X)
        anomalies = np.where(predictions == -1)
        return anomalies[0]
    except Exception as e:
        print(f"Error in anomaly detection: {str(e)}")
        return None

def build_recommendation_system(df, user_col, item_col, rating_col):
    """
    Build a recommendation system using collaborative filtering
    """
    try:
        reader = Reader(rating_scale=(df[rating_col].min(), df[rating_col].max()))
        data = Dataset.load_from_df(df[[user_col, item_col, rating_col]], reader)
        
        # SVD model
        svd_model = SVD(n_factors=100, random_state=42)
        knn_model = KNNBasic(sim_options={'user_based': True})
        
        return {'svd': svd_model, 'knn': knn_model}
    except Exception as e:
        print(f"Error building recommendation system: {str(e)}")
        return None

def assess_fairness(df, protected_attribute, target):
    """
    Assess fairness metrics for the dataset
    """
    try:
        dataset = BinaryLabelDataset(
            df=df,
            label_names=[target],
            protected_attribute_names=[protected_attribute]
        )
        
        metrics = BinaryLabelDatasetMetric(dataset)
        
        return {
            'disparate_impact': metrics.disparate_impact(),
            'statistical_parity_difference': metrics.statistical_parity_difference()
        }
    except Exception as e:
        print(f"Error in fairness assessment: {str(e)}")
        return None

def build_lstm_model(X_train, y_train, n_features):
    """
    Build and train an LSTM model for time series prediction
    """
    try:
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(None, n_features), return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        
        return model
    except Exception as e:
        print(f"Error building LSTM model: {str(e)}")
        return None

def simulate_policy_impact(df, policy_changes, target_variable):
    """
    Simulate the impact of policy changes on target variables
    """
    try:
        simulated_df = df.copy()
        for change in policy_changes:
            if change['type'] == 'increase':
                simulated_df[change['variable']] *= (1 + change['magnitude'])
            elif change['type'] == 'decrease':
                simulated_df[change['variable']] *= (1 - change['magnitude'])
        
        impact = {
            'before_mean': df[target_variable].mean(),
            'after_mean': simulated_df[target_variable].mean(),
            'percent_change': ((simulated_df[target_variable].mean() - df[target_variable].mean()) / 
                             df[target_variable].mean()) * 100
        }
        
        return impact
    except Exception as e:
        print(f"Error in policy simulation: {str(e)}")
        return None