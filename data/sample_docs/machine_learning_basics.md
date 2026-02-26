# Machine Learning Fundamentals

Machine learning (ML) is a subset of artificial intelligence that enables systems to learn
and improve from experience without being explicitly programmed. Instead of following
hard-coded rules, ML algorithms build mathematical models from training data to make
predictions or decisions.

## Types of Machine Learning

### Supervised Learning

In supervised learning, the algorithm learns from labeled training data — input-output
pairs where the correct answer is known. The model learns to map inputs to outputs and
can then generalize to unseen data.

- **Classification**: Predicting a discrete category (e.g., spam vs. not spam, image
  recognition). Common algorithms include Logistic Regression, Decision Trees, Random
  Forests, Support Vector Machines (SVM), and Neural Networks.
- **Regression**: Predicting a continuous value (e.g., house prices, temperature
  forecasting). Common algorithms include Linear Regression, Ridge Regression, Gradient
  Boosted Trees (XGBoost, LightGBM), and Neural Networks.

### Unsupervised Learning

Unsupervised learning works with unlabeled data, finding hidden patterns or structures
without predefined categories.

- **Clustering**: Grouping similar data points together (e.g., customer segmentation).
  Algorithms include K-Means, DBSCAN, and Hierarchical Clustering.
- **Dimensionality Reduction**: Reducing the number of features while preserving
  important information. Techniques include PCA, t-SNE, and UMAP.

### Reinforcement Learning

An agent learns to make decisions by interacting with an environment, receiving rewards
or penalties for its actions. Used in robotics, game playing, and recommendation systems.

## Evaluation Metrics

Choosing the right evaluation metric is critical for assessing model performance:

- **Classification**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Regression**: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean
  Absolute Error (MAE), R-squared (R²)
- **Clustering**: Silhouette Score, Davies-Bouldin Index

## The Machine Learning Workflow

1. **Problem definition** — Clearly define the business question
2. **Data collection** — Gather relevant, representative data
3. **Data preprocessing** — Handle missing values, encode categories, scale features
4. **Feature engineering** — Create informative features from raw data
5. **Model selection** — Choose appropriate algorithms for the problem type
6. **Training** — Fit the model on training data
7. **Evaluation** — Assess performance on held-out test data
8. **Hyperparameter tuning** — Optimize model configuration (Grid Search, Random Search,
   Bayesian Optimization)
9. **Deployment** — Serve the model in production

## Key Concepts

- **Overfitting**: The model memorizes training data but fails on new data. Mitigated by
  regularization, cross-validation, and early stopping.
- **Underfitting**: The model is too simple to capture underlying patterns. Addressed by
  using more complex models or better features.
- **Bias-Variance Tradeoff**: A fundamental tension between a model's ability to fit
  training data (low bias) and generalize to new data (low variance).
