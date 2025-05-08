import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from collections import defaultdict

# Read Excel file
data = pd.read_excel(r"C:\Users...")

# Filter data where Area > 0.00001
filtered_data = data[data['Area'] > 1]

# (Optional) Remove the largest sample by Area, recommended for coastal regions
# max_area_index = filtered_data['Area'].idxmax()  # Get the index of the sample with the largest area
# filtered_data = filtered_data.drop(max_area_index)  # Drop that sample

# Separate features and labels
features = ['ELONGATION', 'RC_CIRCLE', 'COMPLEXITY', 'LINEARITY', 'Area']
X = filtered_data[features]
y = filtered_data['Type']

# Use only samples with type 1 and 2 for training
train_data = filtered_data[filtered_data['Type'] != 0]
X_train = train_data[features]
y_train = train_data['Type']

# Create Random Forest classifier and fit the model
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)

# Get feature importances and sort them
feature_importances = pd.DataFrame({
    'Feature': features,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)
print(feature_importances)

# Calculate splitting thresholds for all features
thresholds = defaultdict(list)
for tree in rf.estimators_:
    tree_features = tree.tree_.feature
    tree_thresholds = tree.tree_.threshold
    for feature, threshold in zip(tree_features, tree_thresholds):
        if feature != -2:
            feature_name = features[feature]
            thresholds[feature_name].append(threshold)

# Output the average splitting threshold for each feature
all_thresholds = {feature: np.mean(thresholds[feature]) for feature in features if feature in thresholds}
print("\nAverage splitting thresholds for all features:")
for feature, threshold in all_thresholds.items():
    print(f"{feature} threshold: {threshold:.4f}")

# Stepwise feature selection based on cross-validation accuracy
cv_scores = []
best_features = []
for i in range(1, len(features) + 1):
    # Select top i important features
    selected_features = feature_importances['Feature'].head(i).values
    X_train_subset = X_train[selected_features]

    # Compute cross-validation accuracy
    scores = cross_val_score(rf, X_train_subset, y_train, cv=10, scoring='accuracy')
    mean_score = scores.mean()
    cv_scores.append(mean_score)
    best_features.append(selected_features)

    print(f"Cross-validation accuracy with top {i} features: {mean_score:.4f}")

# Identify the feature set with the best cross-validation score
optimal_index = np.argmax(cv_scores)
optimal_features = best_features[optimal_index]

print(f"\nOptimal number of features: {len(optimal_features)}")
print(f"Selected features: {optimal_features}")

# Recalculate thresholds using only optimal features
thresholds = defaultdict(list)
for tree in rf.estimators_:
    tree_features = tree.tree_.feature
    tree_thresholds = tree.tree_.threshold
    for feature, threshold in zip(tree_features, tree_thresholds):
        if feature != -2:
            feature_name = features[feature]
            if feature_name in optimal_features:
                thresholds[feature_name].append(threshold)

# Output average splitting threshold for each optimal feature
final_thresholds = {feature: np.mean(thresholds[feature]) for feature in optimal_features}
for feature, threshold in final_thresholds.items():
    print(f"{feature} linear waterbody splitting threshold: {threshold:.4f}")
