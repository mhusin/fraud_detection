# Fraud Detection using Machine Learning Algorithms

This notebook covers model training, hyperparameter tuning, and evaluation to classify fraudulent transactions (Class 1: Fraudulent, Class 0: Not).

## Steps Overview
- **Data Preprocessing**: Prepares the dataset by handling missing values, scaling features, and encoding categorical variables.
- **Model Training**: Trains multiple machine learning models (e.g., Logistic Regression, XGBoost, HistGradient Boosting, LightGBM) on the preprocessed data.
- **Hyperparameter Tuning**: Uses RandomizedSearchCV to optimize model hyperparameters for better performance.
- **Model Evaluation**: Evaluates models using accuracy,F1 score, ROC AUC, Precision-Recall AUC, and balanced accuracy.
- **Out-of-Sample Testing**: Tests both base and tuned models on out-of-sample data to compare performance and generalization.

## Results
The GradientBoosting model achieved a Balanced Accuracy of 75.3%, indicates that the model is treating both classes fairly surpassing the baseline model (Logistic Regression) and other tested models. 

- **Accuracy**: The model correctly predicted 88.92% of all instances. Accuracy can be misleading for this imbalanced dataset. It can be high even if the model mostly predicts the majority class (class 0). 
- **F1 Score**: An F1 score of 0.4975 suggests that the model is performing fairly poorly on the positive class. 
- **ROC AUC**: A score of 0.7771 indicates the model does a good job at separating the two classes, though it is not perfect.
- **Precision-Recall AUC**: A score of 0.6354 is moderate and indicates that while the model is better than random guessing, there's room to improve its performance on the minority class (class 1).
- **Balanced Accuracy**: A score of 0.7526 indicates that the model is treating both classes fairly, but there is still room for improvement.

## Data
- Fraud Data (Fraud_Data.csv): Contains transaction data, including user information, transaction timestamps, and transaction labels (fraudulent or not).
- IP to Country Mapping (IpAddress_to_Country.xlsx): Maps IP addresses to countries, which is used for feature engineering.

## Requirements
- numpy
- pandas
- scikit-learn
- xgboost
- lightgbm
- matplotlib
- seaborn
- imbalanced-learn
- pickle
- os

## fraud_detection.ipynb
- Cell 1: To train the initial models. After training, it prints evaluation metrics and shows visualizations such as the confusion matrix.
  The models are saved as .pkl files in the saved_models folder for future use.
  The trained models are saved as .pkl files in the folder = "saved_models" and can be loaded for future use or deployment. Set the data_path and iptocountry_path.
- Cell 2: To fine-tune the hyperparameters of the models to improve performance. It uses RandomSearchCV to find the best settings and retrains the models with those settings.
  The trained models are saved as .pkl files in the folder = "saved_tuned_models" and can be loaded for future use or deployment. Set the data_path and iptocountry_path.
- Cell 3: To load the trained model, .pkl files and evaluates the best model with visualizations like histogram of predicted probabilities, confusion matrix, ROC curve, precision-Recall curvee
- Cell 4: To test the model on unseen data to see how well it generalizes. It prints evaluation metrics to assess performance on out-of-sample data.
  Set the out_of_sample_path and iptocountry_path

## Next steps
- Explore additional features and feature combinations to capture more patterns.
- Implement a pipeline for preprocessing, feature engineering, and modeling.
- Explore more extensive hyperparameter tuning such as GridSearchCV for the selected models to improve performance.
- Evaluate models with cost-sensitive metrics if fraud detection has varying financial impacts.
