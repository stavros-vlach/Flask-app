import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import pickle

df = pd.read_csv("./Social_Network_Ads.csv")
print(df.head(), "\n")
print(df.info(), "\n")
df = df.drop('User ID', axis=1)
print("Number of NaN values:\n", df.isna().sum(), "\n")

df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

ig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))  
axes = axes.flatten()

for i, v in enumerate(['Age', 'EstimatedSalary']):
    sns.histplot(df[v], ax=axes[i])
    axes[i].set_title(v)
    axes[i].tick_params(axis='x', labelsize=6)

plt.tight_layout()
plt.show()

X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,  random_state=42, stratify=y)

scaler = StandardScaler()
cols_to_scale = [1, 2]


X_train[:, cols_to_scale] = scaler.fit_transform(X_train[:, cols_to_scale])
X_test[:, cols_to_scale] = scaler.transform(X_test[:, cols_to_scale])

model = LogisticRegression()

param_grid = {'C': [0.01, 0.1, 1, 10, 100],      
    'penalty': ['l1', 'l2'],           
    'solver': ['liblinear', 'saga']            
}

grid_search = GridSearchCV(
    estimator=model, 
    param_grid=param_grid, 
    cv=5,                
    scoring='accuracy',  
    n_jobs=-1            
)

grid_search.fit(X_train, y_train)

print("Best parameters found:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

# Use the best estimator to predict
best_model = grid_search.best_estimator_


y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Purchased', 'Purchased'], yticklabels=['Not Purchased', 'Purchased'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

train_accuracy = best_model.score(X_train, y_train)
test_accuracy = best_model.score(X_test, y_test)

print(f"Training Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

with open("logistic_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
    
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
