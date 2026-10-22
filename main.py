import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print("=== 1. ЗАВАНТАЖЕННЯ ТА ОЧИЩЕННЯ ДАНИХ ===")
# Завантаження датасету
df = pd.read_csv('hotel_bookings.csv')

# Вилучення колонок, які викликають витік даних або не мають аналітичного значення
drop_cols = ['reservation_status', 'reservation_status_date', 'name', 'email', 'phone-number', 'credit_card']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])
df = df.drop_duplicates()

# Заповнення пропусків відповідно до змісту змінних
if 'agent' in df.columns:
    df['agent'] = df['agent'].fillna(0)
if 'company' in df.columns:
    df['company'] = df['company'].fillna(0)
if 'children' in df.columns:
    df['children'] = df['children'].fillna(0)
if 'country' in df.columns:
    df['country'] = df['country'].fillna(df['country'].mode()[0])

print(f"Розмірність очищеного набору даних: {df.shape}")

print("\n=== 2. АНАЛІЗ АНОМАЛІЙ ЗА МЕТОДОМ IQR ===")
num_features = ['lead_time', 'adr', 'stays_in_week_nights', 'adults', 'previous_cancellations']
for col in num_features:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        print(f"Ознака {col}: виявлено {outliers} потенційних аномалій (IQR)")

print("\n=== 3. НАВЧАННЯ ТА ОЦІНКА МОДЕЛЕЙ ===")
X = df.drop(columns=['is_canceled'])
y = df['is_canceled']

cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Стратифікований поділ вибірки (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Побудова конвеєра обробки ознак
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ]
)

# Три класичні алгоритми класифікації
models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42), "model_logistic_regression.pkl"),
    "Decision Tree": (DecisionTreeClassifier(max_depth=10, random_state=42), "model_decision_tree.pkl"),
    "K-NN": (KNeighborsClassifier(n_neighbors=5), "model_knn.pkl")
}

results = []

for name, (model, filename) in models.items():
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0
    
    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4)
    })
    
    # Серіалізація моделі у формат pickle
    with open(filename, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Модель '{name}' збережено у файл {filename}")

print("\n=== ПОРІВНЯЛЬНА ТАБЛИЦЯ МЕТРИК ===")
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))