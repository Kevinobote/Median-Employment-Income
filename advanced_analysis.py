import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_classif
import shap
import lime
import lime.lime_tabular

def analyze_demographic_trends(df, demographic_cols, target_col):
    """
    Analyze trends across demographic variables
    """
    try:
        trends = {}
        for col in demographic_cols:
            # Calculate aggregate statistics
            agg_stats = df.groupby(col)[target_col].agg(['mean', 'median', 'std', 'count'])
            trends[col] = agg_stats

            # Visualization
            plt.figure(figsize=(12, 6))
            sns.boxplot(x=col, y=target_col, data=df)
            plt.xticks(rotation=45)
            plt.title(f'{target_col} Distribution by {col}')
            plt.tight_layout()
            plt.show()

        return trends
    except Exception as e:
        print(f"Error in demographic analysis: {str(e)}")
        return None

def visualize_correlations(df, method='pearson', figsize=(12, 8)):
    """
    Create enhanced correlation visualizations
    """
    try:
        # Calculate correlations
        corr_matrix = df.corr(method=method)
        
        # Plot correlation matrix
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix), k=1)
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlations')
        plt.tight_layout()
        plt.show()
        
        return corr_matrix
    except Exception as e:
        print(f"Error in correlation visualization: {str(e)}")
        return None

def perform_feature_selection(X, y, method='mutual_info', k=10):
    """
    Perform feature selection using various methods
    """
    try:
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        else:  # regression
            selector = SelectKBest(score_func=f_regression, k=k)
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Plot feature importance scores
        plt.figure(figsize=(10, 6))
        scores = selector.scores_
        plt.bar(X.columns, scores)
        plt.xticks(rotation=45)
        plt.title('Feature Importance Scores')
        plt.tight_layout()
        plt.show()
        
        return selected_features, X_selected
    except Exception as e:
        print(f"Error in feature selection: {str(e)}")
        return None, None

def explain_predictions(model, X_train, X_test, feature_names):
    """
    Generate model explanations using SHAP and LIME
    """
    try:
        # SHAP explanations
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # SHAP summary plot
        plt.figure()
        shap.summary_plot(shap_values, X_test, feature_names=feature_names)
        plt.tight_layout()
        plt.show()
        
        # LIME explanation for a single instance
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=feature_names,
            class_names=['Output'],
            mode='regression'
        )
        exp = lime_explainer.explain_instance(
            X_test.iloc[0].values, 
            model.predict,
            num_features=10
        )
        exp.show_in_notebook()
        
        return {'shap_values': shap_values, 'lime_explainer': lime_explainer}
    except Exception as e:
        print(f"Error in generating explanations: {str(e)}")
        return None

def evaluate_model_performance(y_true, y_pred, model_type='regression'):
    """
    Comprehensive model evaluation with visualizations
    """
    try:
        results = {}
        if model_type == 'regression':
            # Calculate metrics
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            
            results = {
                'mse': mse,
                'rmse': rmse,
                'r2': r2
            }
            
            # Residual plot
            plt.figure(figsize=(10, 6))
            residuals = y_true - y_pred
            sns.scatterplot(x=y_pred, y=residuals)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title('Residual Plot')
            plt.tight_layout()
            plt.show()
            
        else:  # classification
            # ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(10, 6))
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.tight_layout()
            plt.show()
            
            # Classification report
            results = classification_report(y_true, y_pred, output_dict=True)
        
        return results
    except Exception as e:
        print(f"Error in model evaluation: {str(e)}")
        return None