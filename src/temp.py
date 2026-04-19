from preprocess import preprocess_pipeline

X_train, X_test, y_train, y_test = preprocess_pipeline("data/real.csv")

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Target distribution:\n", y_train.value_counts())