import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm
from libsvm import svmutil


data = pd.read_csv("winequality-complete.csv")

X = data.drop("quality", axis=1)
y = data["quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# depth_range = range(1, 21)
# cv_scores = []
# for depth in depth_range:
#     model = DecisionTreeRegressor(max_depth=depth, random_state=0)
#     scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
#     cv_scores.append(np.mean(-scores))

# plt.figure(figsize=(10, 6))
# plt.plot(depth_range, cv_scores, marker='o')
# plt.title('Cross-Validation Scores for Different Tree Depths')
# plt.xlabel('Tree Depth')
# plt.ylabel('Negative Mean Squared Error (CV Score)')
# plt.grid(True)
# plt.show()

# optimal_depth = depth_range[np.argmin(cv_scores)]
# print(f"Optimal Tree Depth: {optimal_depth}")
# optimal_model = DecisionTreeRegressor(max_depth=optimal_depth, random_state=0)
# optimal_model.fit(X_train, y_train)

# plt.figure(figsize=(15, 10))
# plot_tree(optimal_model, filled=True, feature_names=X.columns, rounded=True)
# plt.show()

# y_pred = optimal_model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error on Test Set: {mse}")

# def train_random_forest(n_estimators, max_features):
#     clf = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, random_state=0)
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     return mse

# n_estimators_values = [25, 500]
# max_features_values = np.arange(1, X.shape[1] + 1)
# plt.figure(figsize=(10, 6))
# for n_estimators in n_estimators_values:
#     mse_errors = []
#     for max_features in max_features_values:
#         error = train_random_forest(n_estimators, max_features)
#         mse_errors.append(error)
#     plt.plot(max_features_values, mse_errors, label=f'{n_estimators} Trees')
# plt.xlabel('Number of Predictors')
# plt.ylabel('MSE')
# plt.title('Random Forest MSE for Different Number of Trees and Predictors')
# plt.legend()
# plt.show()


# Standardize the data before applying PCA
scaler_pca = StandardScaler()
X_train_scaled_pca = scaler_pca.fit_transform(X_train)
X_test_scaled_pca = scaler_pca.transform(X_test)

# Apply PCA to identify the most important components
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled_pca)
X_test_pca = pca.transform(X_test_scaled_pca)

# Convert labels to 1 and -1 for LIBSVM
y_train_libsvm = 2 * (y_train / 10) - 1
print("Unique values in y_train:", np.unique(y_train))
print("Unique values in y_train_libsvm:", np.unique(y_train_libsvm))
print("Sample X_train_pca:", X_train_pca[:5])
print("Sample y_train_libsvm:", y_train_libsvm[:5])

# Train SVM models with different kernels using LIBSVM (SMO solver)
linear_svm_pca = svmutil.svm_train(y_train_libsvm.tolist(), X_train_pca.tolist(), '-t 0')
radial_svm_pca = svmutil.svm_train(y_train_libsvm.tolist(), X_train_pca.tolist(), '-t 2')
polynomial_svm_pca = svmutil.svm_train(y_train_libsvm.tolist(), X_train_pca.tolist(), '-t 1')

# Create a meshgrid for contourf
h = .02  # Step size in the mesh
x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Plot decision boundaries with contourf
def plot_contourf(ax, model, title):
    meshgrid = np.c_[xx.ravel(), yy.ravel()]
    p_label, _, _ = svmutil.svm_predict([0] * len(meshgrid), meshgrid.tolist(), model)
    Z = np.array(p_label)
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, edgecolors='k', cmap=plt.cm.Paired)
    ax.set_title(title)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')

# Calculate and print Mean Squared Error (MSE) for each model
p_label, _, _ = svmutil.svm_predict(y_test.tolist(), X_test_pca.tolist(), linear_svm_pca)
mse_linear_pca = mean_squared_error(y_test, p_label)

p_label, _, _ = svmutil.svm_predict(y_test.tolist(), X_test_pca.tolist(), radial_svm_pca)
mse_radial_pca = mean_squared_error(y_test, p_label)

p_label, _, _ = svmutil.svm_predict(y_test.tolist(), X_test_pca.tolist(), polynomial_svm_pca)
mse_poly_pca = mean_squared_error(y_test, p_label)

# Display MSE values
print("MSE for Linear Kernel with PCA:", mse_linear_pca)
print("MSE for Radial Basis Kernel with PCA:", mse_radial_pca)
print("MSE for Polynomial Kernel with PCA:", mse_poly_pca)

# Visualize the results with contourf
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Linear Kernel
plot_contourf(axs[0], linear_svm_pca, f'Linear SVM with PCA Components\nMSE: {mse_linear_pca:.4f}')

# Radial Basis Kernel
plot_contourf(axs[1], radial_svm_pca, f'Radial SVM with PCA Components\nMSE: {mse_radial_pca:.4f}')

# Polynomial Kernel
plot_contourf(axs[2], polynomial_svm_pca, f'Polynomial SVM with PCA Components\nMSE: {mse_poly_pca:.4f}')

plt.tight_layout()
plt.show()