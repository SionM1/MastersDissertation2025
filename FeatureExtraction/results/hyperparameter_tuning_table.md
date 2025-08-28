# Hyperparameter Tuning Results

## Table 1: Model Performance Comparison

| Rank | Model | F1-Score | AUC | Precision | Recall | Training Time |
|------|-------|----------|-----|-----------|---------|---------------|
| 1 | Autoencoder | 0.9904 | 0.9907 | 0.9978 | 0.9830 | 3.693s |
| 2 | LOF | 0.9896 | 0.9906 | 0.9962 | 0.9830 | 0.100s |
| 3 | EllipticEnvelope | 0.9883 | 0.9914 | 0.9930 | 0.9836 | 0.176s |
| 4 | IsolationForest | 0.9883 | 0.9777 | 0.9930 | 0.9836 | 0.191s |
| 5 | OneClassSVM | 0.9883 | 0.9904 | 0.9946 | 0.9820 | 0.077s |

## Table 2: Optimal Hyperparameters

| Model | Optimal Parameters |
|-------|--------------------|
| Autoencoder | epochs=50, latent_dim=8, dropout_rate=0.0 |
| LOF | n_neighbors=20, contamination=0.1 |
| EllipticEnvelope | support_fraction=0.8, contamination=0.1 |
| IsolationForest | n_estimators=100, max_samples=0.8, contamination=0.1 |
| OneClassSVM | nu=0.05, kernel=rbf, gamma=scale |
