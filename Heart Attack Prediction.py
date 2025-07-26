#!/usr/bin/env python
# coding: utf-8

# **Load the required libraries**

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import janitor
import warnings
warnings.filterwarnings('ignore')


# **Load the dataset**

# In[2]:


df = pd.read_csv("C:/Users/ADMIN/Desktop/Data Science/Datasets/Datasets/Heart_Attack_Risk_Levels_Dataset.csv").clean_names()


# **Inspect the dataset**

# In[3]:


df.head(10)


# **Last few observations of the dataset**

# In[13]:


df.tail(10)


# **Data Types Check**

# In[15]:


df.dtypes


# **Structure of the dataset**

# In[14]:


df.info()


# **Unique Values and Cardinality**

# In[16]:


df.nunique()


# **Checking for missing values**

# In[5]:


df.isnull().sum()


# **Data type distribution**

# In[6]:


sns.countplot(y=df.dtypes ,data=df)
plt.title("Data type Distribution")
plt.xlabel("count of each data type")
plt.ylabel("data types")
plt.show()


# **Check for duplicates**

# In[7]:


df.duplicated().sum()


# **Data Cleaning & Preprocessing**
# 
# **Drop unwanted columns**

# In[8]:


df = df.drop(columns=['risk_level', 'recommendation'])


# **Label encoding of the target variable**

# In[10]:


## Initialize LabelEncoder
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()

# Apply label encoding to the 'Result' column
df['result'] = label.fit_transform(df['result'])


# **Exploratory Data Analysis**
# 
# **Summary Statistics**

# In[11]:


df.describe()


# **Distribution of Numerical Variables**

# In[45]:


numeric_col = df.drop(columns = ["result", "gender"])
for col in numeric_col:
    plt.figure(figsize = (8, 4))
    sns.histplot(df[col], kde = True, bins = 15, color = "blue")
    plt.title(f'Histogram and KDE for {col}')
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()


# **Boxplots to Detect Outliers**

# In[18]:


for col in numeric_col:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()


# **Gender Distribution**

# In[19]:


sns.countplot(x="gender", data=df)
plt.title('Gender Distribution')
plt.ylabel("Frequency")
plt.xlabel("Gender")
plt.xticks([0, 1], labels=["Female", "Male"])
plt.show()


# **Distribution of the target variable**

# In[20]:


df["result"].value_counts()


# In[21]:


## Class Distribution
sns.countplot(x="result", data=df)
plt.title('Heart attack distribution')
plt.ylabel("Frequency")
plt.xticks([0, 1], labels=["Negative", "Positive"])
plt.xlabel("Heart attack")
plt.show()


# **Correlation Matrix**

# In[22]:


corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()


# **Group-wise Analysis (e.g., Risk Level vs Numeric Variables)**

# In[25]:


for col in numeric_col:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='result', y=col, data=df)
    plt.title(f'{col} vs Heart Attack Outcome')
    plt.show()


# **Bivariate Relationships (Scatterplots)**

# In[28]:


sns.pairplot(numeric_col)
plt.suptitle("Scatterplot Matrix", y=1.02)
plt.show()


# **Cross Tabulation and Stacked Bar Charts**

# In[44]:


## Load required modules
import matplotlib.pyplot as plt
import pandas as pd

## Create a copy of your DataFrame
df_plot = df.copy()

## Rename values for clarity
df_plot['result'] = df_plot['result'].replace({1 : 'Heart Attack', 0 : 'No Heart Attack'})
df_plot['gender'] = df_plot['gender'].replace({0: 'Female', 1: 'Male'})

## Create normalized crosstab
crosstab = pd.crosstab(df_plot['result'], df_plot['gender'], normalize='index')

## Plot
crosstab.plot(kind='bar', stacked=True)

## Set labels and title
plt.title("Gender Proportion by Heart Attack Outcome")
plt.ylabel("Proportion")
plt.xlabel("Heart Attack Outcome")
plt.legend(title="Gender")
plt.tight_layout()
plt.show()


# **Chi-square Tests for Categorical Associations**

# In[34]:


## Load required module
from scipy.stats import chi2_contingency
import pandas as pd

## Create contingency table
table = pd.crosstab(df['gender'], df['result'])

## Perform Chi-square test
chi2, p, dof, ex = chi2_contingency(table)

## Print result
print(f"Chi-square test between Gender and Heart Attack Outcome: p-value = {p:.4f}")


# **T-tests (Numeric vs. Categorical)**

# In[40]:


## Load required module
from scipy.stats import ttest_ind

## Get all numeric columns
numeric_col = df.drop(columns = ["result", "gender"])

## Split the dataset into two groups based on 'Result'
group1 = df[df['result'] == 1]
group2 = df[df['result'] == 0]

## Perform independent t-test for each numeric column
ttest_results = []
for col in numeric_col:
    t_stat, p_val = ttest_ind(group1[col], group2[col], equal_var=False)  # Welch's t-test
    ttest_results.append((col, p_val))

ttest_results


# **Defining the X and y features**

# In[46]:


X = df.drop(columns = ["result"])
y = df["result"]


# **Splitting data into training and test set**

# In[47]:


## Load the required module
from sklearn.model_selection import train_test_split

## Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 42, test_size = 0.2)


# In[48]:


print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# **Feature Scaling/Standardization**

# In[49]:


## Load the required module
from sklearn.preprocessing import MinMaxScaler

## Initialize the scaler
scaler = MinMaxScaler()

## Fit the scaler
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# **Model Training**
# 
# **1. Logistic Regression**

# In[50]:


## Load required modules
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

## Initialize the model with better configuration
log = LogisticRegression(random_state=42, solver='lbfgs', max_iter=1000)

## Fit the model
log.fit(X_train, y_train)

## Make predictions
log_pred = log.predict(X_test)
log_score = accuracy_score(y_test, log_pred)

## Evaluate performance
print("Classification Report:\n", classification_report(y_test, log_pred))
print("F1 Score:", f1_score(y_test, log_pred, average='macro'))
print("Precision:", precision_score(y_test, log_pred, average='macro'))
print("Recall:", recall_score(y_test, log_pred, average='macro'))
print("Accuracy:", accuracy_score(y_test, log_pred))

## Confusion Matrix
cm = confusion_matrix(y_test, log_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# **Decision Trees**

# In[52]:


## Load required modules
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt

## Initialize model
dt = DecisionTreeClassifier(random_state=42)

## Define a more efficient parameter grid
parameters = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'splitter': ['best']
}

## Grid Search
grid_search_dt = GridSearchCV(dt, parameters, cv=5, n_jobs=-1, verbose=1, scoring='f1_macro')
grid_search_dt.fit(X_train, y_train)

## Best model prediction
dt_pred = grid_search_dt.predict(X_test)
dt_score = accuracy_score(y_test, dt_pred)

## Evaluation
print("Best Parameters Found:", grid_search_dt.best_params_)
print("Classification Report:\n", classification_report(y_test, dt_pred))
print("F1 Score:", f1_score(y_test, dt_pred, average='macro'))
print("Precision:", precision_score(y_test, dt_pred, average='macro'))
print("Recall:", recall_score(y_test, dt_pred, average='macro'))
print("Accuracy:", accuracy_score(y_test, dt_pred))


# **2. Random Forest**

# In[53]:


## Import libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt

## Define your model
rf = RandomForestClassifier(class_weight='balanced', random_state=42)

## Define a reduced hyperparameter grid
param_dist = {
    'n_estimators': [100, 300, 500, 800],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

## Cross-validation strategy
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)

## Use RandomizedSearchCV for efficiency
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=20,  # Try only 20 random combinations
    cv=cv,
    scoring='f1_macro',  # Better than accuracy for imbalanced data
    n_jobs=-1,
    verbose=2,
    random_state=42
)

## Fit the model
best_model = random_search.fit(X_train, y_train)

## Make predictions
rf_pred = best_model.predict(X_test)
rf_score = accuracy_score(y_test, rf_pred)

## Evaluate performance
print("Best Parameters:", random_search.best_params_)
print("Best Cross-Validated F1 Score:", random_search.best_score_)
print("\n Classification Report:\n", classification_report(y_test, rf_pred))
print("Accuracy Score:", accuracy_score(y_test, rf_pred))
print("F1 Score:", f1_score(y_test, rf_pred, average='macro'))
print("Precision:", precision_score(y_test, rf_pred, average='macro'))
print("Recall:", recall_score(y_test, rf_pred, average='macro'))

## Plot the confusion matrix
cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# **Support Vector Machines**

# In[54]:


## Import required libraries
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, accuracy_score)

## Define CV strategy
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

## Common SVM model with probability enabled
svm = SVC(probability=True, random_state=42)

## Define separate hyperparameter grids
# RBF Kernel
param_grid_rbf = {
    'kernel': ['rbf'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'shrinking': [True, False]
}

# Poly Kernel
param_grid_poly = {
    'kernel': ['poly'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3],
    'coef0': [0.0, 0.5],
    'shrinking': [True, False]
}

# Sigmoid Kernel
param_grid_sigmoid = {
    'kernel': ['sigmoid'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'coef0': [0.0, 0.5],
    'shrinking': [True, False]
}

## Combine the grids
param_grid_combined = [
    param_grid_rbf,
    param_grid_poly,
    param_grid_sigmoid
]

## Use RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=svm,
    param_distributions=param_grid_combined,
    n_iter=20,  # Try 20 random combinations
    scoring='f1_macro',
    n_jobs=-1,
    cv=cv,
    verbose=2,
    random_state=42
)

## Train the model
best_model = random_search.fit(X_train, y_train)

## Make predictions
svm_pred = best_model.predict(X_test)
svm_score = accuracy_score(y_test, svm_pred)

## Evaluate performance
print("Best Parameters:", best_model.best_params_)
print("Accuracy:", accuracy_score(y_test, svm_pred))
print("F1 Score (macro):", f1_score(y_test, svm_pred, average='macro'))
print("Precision (macro):", precision_score(y_test, svm_pred, average='macro'))
print("Recall (macro):", recall_score(y_test, svm_pred, average='macro'))
print("\n Classification Report:\n", classification_report(y_test, svm_pred))

## onfusion matrix visualization
cm = confusion_matrix(y_test, svm_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
plt.title("Confusion Matrix - SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# **XGBoost Classifier**

# In[55]:


## Load required modules
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, 
                             f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns

## Define the parameter grid
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 1],
    'min_child_weight': [1, 5],
    'scale_pos_weight': [1]  
}

## Define cross-validation strategy
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

## Initialize the model
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

## Grid Search
grid_search = GridSearchCV(estimator=xgb,
                           param_grid=param_grid,
                           scoring='f1_macro',
                           cv=cv,
                           n_jobs=-1,
                           verbose=1)

## Fit the model
grid_result = grid_search.fit(X_train, y_train)

## Best estimator
best_xgb = grid_result.best_estimator_

## Predict class labels
xgb_pred = best_xgb.predict(X_test)

## Predict probabilities
xgb_prob = best_xgb.predict_proba(X_test)
xgb_score = accuracy_score(y_test, xgb_pred)

## Get probabilities of the positive class (label = 1)
xgb_prob_positive = xgb_prob[:, 1]

## Evaluate performance
print(" Best Hyperparameters:", grid_result.best_params_)
print("\n Classification Report:\n", classification_report(y_test, xgb_pred))
print(" F1 Score (macro):", f1_score(y_test, xgb_pred, average='macro'))
print(" Precision (macro):", precision_score(y_test, xgb_pred, average='macro'))
print(" Recall (macro):", recall_score(y_test, xgb_pred, average='macro'))
print(" Accuracy:", accuracy_score(y_test, xgb_pred))

## Print some of the predicted probabilities
print("\nSample predicted probabilities:\n", xgb_prob[:10])

## Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, xgb_pred), annot=True, fmt='d', cmap='YlOrBr')
plt.title("Confusion Matrix - XGBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

## ROC Curve 
if len(set(y_test)) == 2:
    fpr, tpr, thresholds = roc_curve(y_test, xgb_prob_positive)
    auc_score = roc_auc_score(y_test, xgb_prob_positive)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - XGBoost")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


# **K Nearest Neighbors**

# In[56]:


## Load the required libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)

## Define the pipeline 
pipeline = Pipeline([
    ('knn', KNeighborsClassifier())
])

## Define hyperparameters to tune
param_grid = {
    'knn__n_neighbors': range(15, 25),
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan']
}

## Define cross-validation strategy
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

## Setup Grid Search
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    n_jobs=-1,
    cv=cv,
    scoring='f1_macro',   
    error_score=0,
    verbose=1
)

## Fit the model
best_model = grid_search.fit(X_train, y_train)

## Make predictions
knn_pred = best_model.predict(X_test)
knn_score = accuracy_score(y_test, knn_pred)

## Display best hyperparameters
print("Best Hyperparameters:\n", grid_search.best_params_)
print("Best Cross-Validated F1 Macro Score:\n", grid_search.best_score_)

## Classification metrics
print("\n📊 Classification Report:\n", classification_report(y_test, knn_pred))
print("Accuracy Score:", accuracy_score(y_test, knn_pred))
print("F1 Score (Macro):", f1_score(y_test, knn_pred, average='macro'))
print("Precision (Macro):", precision_score(y_test, knn_pred, average='macro'))
print("Recall (Macro):", recall_score(y_test, knn_pred, average='macro'))

## Confusion matrix plot
cm = confusion_matrix(y_test, knn_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("🔍 Confusion Matrix - KNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# **Gradient Boosting Machines**

# In[57]:


## Import required modules
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

## Define the model
gbm = GradientBoostingClassifier(random_state=42)

## Define hyperparameter space
param_dist = {
    'n_estimators': np.arange(80, 201, 20),         # 80 to 200 in steps of 20
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'max_depth': [3, 4, 5, 6],
    'subsample': [0.6, 0.8, 1.0],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

## Cross-validation strategy
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

## Setup RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=gbm,
                                   param_distributions=param_dist,
                                   n_iter=20,                  # try only 20 random combinations
                                   scoring='f1',
                                   n_jobs=-1,
                                   cv=cv,
                                   verbose=1,
                                   random_state=42)

## Fit the model
best_model = random_search.fit(X_train, y_train)

## Predict on test data
gbm_pred = best_model.predict(X_test)
gbm_score = accuracy_score(y_test, gbm_pred)

## Print best hyperparameters and CV score
print("Best Parameters:\n", random_search.best_params_)
print("Best CV F1 Score:\n", random_search.best_score_)

## Evaluation metrics
print("\nClassification Report:\n", classification_report(y_test, gbm_pred))
print("Accuracy Score:", accuracy_score(y_test, gbm_pred))
print("F1 Score:", f1_score(y_test, gbm_pred, average='macro'))
print("Precision:", precision_score(y_test, gbm_pred, average='macro'))
print("Recall:", recall_score(y_test, gbm_pred, average='macro'))

## Confusion Matrix Plot
cm = confusion_matrix(y_test, gbm_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
plt.title("Confusion Matrix - Gradient Boosting (Random Search)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# **Ada Boost Classifier**

# In[58]:


## Import required modules
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import numpy as np

## Define base estimator
base_estimator = DecisionTreeClassifier(max_depth=1)

## Define AdaBoost model
ada = AdaBoostClassifier(estimator=base_estimator, n_estimators=180, learning_rate=1.0)

## Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 180, 250],
    'learning_rate': [0.01, 0.1, 1.0],
    'estimator__max_depth': [1, 2, 3]
}

grid_search = GridSearchCV(ada, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
grid_search.fit(X_train, y_train)

## Best model
best_ada = grid_search.best_estimator_
print("Best Parameters from Grid Search:", grid_search.best_params_)

## Cross-validation scores
cv_scores = cross_val_score(best_ada, X_train, y_train, cv=5, scoring='f1_macro')
print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

## Fit model on full training set
best_ada.fit(X_train, y_train)

## Predict on test set
ada_pred = best_ada.predict(X_test)
ada_score = accuracy_score(y_test, ada_pred)

## Evaluate model
print("\n=== Evaluation on Test Set ===")
print("Classification Report:\n", classification_report(y_test, ada_pred))
print("Accuracy:", accuracy_score(y_test, ada_pred))
print("F1 Score:", f1_score(y_test, ada_pred, average='macro'))
print("Precision:", precision_score(y_test, ada_pred, average='macro'))
print("Recall:", recall_score(y_test, ada_pred, average='macro'))

## Confusion Matrix
cm = confusion_matrix(y_test, ada_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

## Feature Importances
importances = best_ada.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(np.arange(len(importances)), importances)
plt.title("Feature Importances")
plt.xlabel("Feature Index")
plt.ylabel("Importance Score")
plt.grid(True)
plt.tight_layout()
plt.show()


# **Voting Classifier**

# In[59]:


## Load required libraries
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.model_selection import RepeatedStratifiedKFold

## Define individual base models
log_clf = LogisticRegression(solver='liblinear', random_state=42)
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
ada_clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=100,
    learning_rate=0.5,
    random_state=42
)

## Combine them into a Voting Classifier (soft voting)
voting_clf = VotingClassifier(
    estimators=[
        ('lr', log_clf),
        ('rf', rf_clf),
        ('ada', ada_clf)
    ],
    voting='soft',  # soft = uses probabilities
    n_jobs=-1
)

## Fit the model
voting_clf.fit(X_train, y_train)

## Make predictions
vote_pred = voting_clf.predict(X_test)
vote_score = accuracy_score(y_test, vote_pred)

## Evaluate
print("Classification Report:\n", classification_report(y_test, vote_pred))
print("Accuracy:", accuracy_score(y_test, vote_pred))
print("F1 Score:", f1_score(y_test, vote_pred, average='macro'))
print("Precision:", precision_score(y_test, vote_pred, average='macro'))
print("Recall:", recall_score(y_test, vote_pred, average='macro'))

## Confusion matrix
cm = confusion_matrix(y_test, vote_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='BuPu')
plt.title("Confusion Matrix - Voting Classifier")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# **Stacking Classifier**

# In[60]:


## Import required libraries
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

## Define base models
base_learners = [
    ('logreg', LogisticRegression(max_iter=1000)),
    ('dt', DecisionTreeClassifier()),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svc', SVC(C=10, kernel='rbf', gamma='scale', probability=True)),  # SVM needs probability=True for stacking
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
]

## Define the meta-model
meta_learner = LogisticRegression()

## Define the stacking classifier
stack_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5,
    n_jobs=-1,
    passthrough=False
)

## Fit the model
stack_model.fit(X_train, y_train)

## Make predictions
y_pred = stack_model.predict(X_test)
stack_score = accuracy_score(y_test, y_pred)

## Evaluate performance
print("STACKING MODEL PERFORMANCE")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))

## Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='YlGnBu')
plt.title("Confusion Matrix - Stacking Model")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# **Stochastic Gradient Descent**

# In[61]:


## Load the required modules
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)

## Initialize the model
sgd = SGDClassifier(random_state=42)

## Define the parameters to tune
param_grid = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
    'loss': ['hinge', 'log_loss'],  # 'log_loss' for logistic regression
    'penalty': ['l1', 'l2']
}

## Setup Grid Search
grid_search = GridSearchCV(estimator=sgd,
                           param_grid=param_grid,
                           cv=10,
                           scoring='f1_macro',  # Better for multi-class or imbalanced data
                           n_jobs=-1,
                           verbose=1)

## Fit the model
grid_search.fit(X_train, y_train)

## Make Predictions
sgd_pred = grid_search.predict(X_test)
sgd_score = accuracy_score(y_test, sgd_pred)

## Best parameters and CV score
print("Best Parameters:", grid_search.best_params_)
print("Best CV F1 Macro Score:", grid_search.best_score_)

## Performance evaluation
print("\n📊 Classification Report:\n", classification_report(y_test, sgd_pred))
print("Accuracy:", accuracy_score(y_test, sgd_pred))
print("F1 Score (Macro):", f1_score(y_test, sgd_pred, average='macro'))
print("Precision (Macro):", precision_score(y_test, sgd_pred, average='macro'))
print("Recall (Macro):", recall_score(y_test, sgd_pred, average='macro'))

## Confusion Matrix
cm = confusion_matrix(y_test, sgd_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
plt.title("Confusion Matrix - SGD Classifier")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# **Model Comparison**

# In[62]:


models = pd.DataFrame({
    'Model': ['Logistic Regression', "Decision Trees", 'Random Forest Classifier', 'SVM', "XG Boost", 'KNN', "Gradient Boosting", "Ada Boost Classifier", 'Voting Classifier', 'Stacking Classifier',
             'Stachastic Gradient Boosting'],
    'Score': [log_score, dt_score, rf_score, svm_score, xgb_score, knn_score, gbm_score, ada_score, vote_score, stack_score, sgd_score]
})

models.sort_values(by = 'Score', ascending = False)


# **Best model - XGBoost**

# **SHAP Values for Deeper Interpretability**

# In[63]:


feature_names = [
    'age',
    'gender',
    'heart_rate',
    'systolic_blood_pressure',
    'diastolic_blood_pressure',
    'blood_sugar',
    'ck_mb',
    'troponin'
]

import shap

# Create SHAP explainer for XGBoost
explainer = shap.Explainer(best_xgb)

# Compute SHAP values
shap_values = explainer(X_test)

# Plot SHAP summary (global feature importance)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)


# **Bar Chart**

# In[64]:


import pandas as pd
import shap

# Step 1: Define feature names
feature_names = [
    'age',
    'gender',
    'heart_rate',
    'systolic_blood_pressure',
    'diastolic_blood_pressure',
    'blood_sugar',
    'ck_mb',
    'troponin'
]

# Step 2: Convert X_test to a DataFrame with column names
X_test_df = pd.DataFrame(X_test, columns=feature_names)

# Step 3: Create SHAP explainer and compute values
explainer = shap.Explainer(best_xgb)
shap_values = explainer(X_test_df)

# Step 4: Plot SHAP bar chart with proper feature names
shap.plots.bar(shap_values, max_display=8)


# In[41]:


# Save the best model to a file
import pickle
with open('heart_attack_xgb_model.pkl', 'wb') as f:
    pickle.dump(best_xgb, f)

