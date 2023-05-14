import pandas as pd
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# Load dataset
dataset = pd.read_csv("data/milk.csv")

# Split atribut dan label
X = dataset.drop('Grade', axis=1)
y = dataset['Grade']

# Hold-out Method (70%-30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=49)

# Klasifikasi Naïve Bayes pada Hold-out Method
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_holdout = nb.predict(X_test)
accuracy_holdout = accuracy_score(y_test, y_pred_holdout)
print("1. Akurasi Hold-out Method:", accuracy_holdout)

# K-Fold (k=10)
kf = KFold(n_splits=10)
accuracy_kfold = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    nb.fit(X_train, y_train)
    y_pred_kfold = nb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_kfold)
    accuracy_kfold.append(accuracy)

mean_accuracy_kfold = sum(accuracy_kfold) / len(accuracy_kfold)
print("2a. Akurasi K-Fold (k=10):", mean_accuracy_kfold)

# LOO (Leave-One-Out)
loo = LeaveOneOut()
accuracy_loo = []

for train_index, test_index in loo.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    nb.fit(X_train, y_train)
    y_pred_loo = nb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_loo)
    accuracy_loo.append(accuracy)

mean_accuracy_loo = sum(accuracy_loo) / len(accuracy_loo)
print("2c. Akurasi LOO:", mean_accuracy_loo)

# Normalisasi data dengan MinMaxScaler
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Klasifikasi Naïve Bayes pada data yang telah dinormalisasi
nb.fit(X_train_normalized, y_train)
y_pred_normalized = nb.predict(X_test_normalized)
accuracy_normalized = accuracy_score(y_test, y_pred_normalized)
print("4. Akurasi Naïve Bayes (dengan normalisasi):", accuracy_normalized)

# Klasifikasi Naïve Bayes pada Hold-out Method tanpa normalisasi
nb.fit(X_train, y_train)
y_pred_holdout_nonnormalized = nb.predict(X_test)
accuracy_holdout_nonnormalized = accuracy_score(y_test, y_pred_holdout_nonnormalized)
print("8. Akurasi Hold-out Method (tanpa normalisasi):", accuracy_holdout_nonnormalized)

# Perbandingan akurasi Hold-out Method dengan dan tanpa normalisasi
print("Perbandingan akurasi Hold-out Method:")
print("- Dengan normalisasi:", accuracy_holdout)
print("- Tanpa normalisasi:", accuracy_holdout_nonnormalized)
