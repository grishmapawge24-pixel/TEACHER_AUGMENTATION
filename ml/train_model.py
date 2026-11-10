# =============================================================
# train_model.py
# PURPOSE: Train the Student Performance Prediction ML Model
#          and save it to disk so Flask can use it later.
#
# HOW TO RUN: python ml/train_model.py
# =============================================================

# --- Import the tools we need ---
import pandas as pd          # pandas helps us read CSV files (like Excel for Python)
import numpy as np           # numpy helps with math/numbers
import joblib                # joblib saves our trained model to a file
import os                    # os helps us create folders

# scikit-learn tools:
from sklearn.ensemble import RandomForestRegressor  # The ML algorithm we'll use
from sklearn.model_selection import train_test_split # Splits data into train/test
from sklearn.preprocessing import StandardScaler     # Normalizes our numbers
from sklearn.metrics import mean_absolute_error, r2_score  # Measures accuracy

print("=" * 50)
print("  STUDENT PERFORMANCE MODEL - TRAINING SCRIPT")
print("=" * 50)

# -------------------------------------------------------
# STEP 1: Load the data from our CSV file
# -------------------------------------------------------
print("\n[1/5] Loading data from CSV...")

# pd.read_csv reads the CSV file and stores it as a "DataFrame" (like a table)
df = pd.read_csv("data/student_data.csv")

print(f"      ✅ Loaded {len(df)} student records")
print(f"      Columns: {list(df.columns)}")
print(f"\n      First 3 rows preview:")
print(df.head(3).to_string())

# -------------------------------------------------------
# STEP 2: Separate Inputs (X) and Output (y)
# -------------------------------------------------------
print("\n[2/5] Preparing features (inputs) and target (output)...")

# X = the input columns (what we give the model to make predictions)
# These are called "features" in ML
X = df[["attendance", "study_hours", "previous_marks", "assignments_completed"]]

# y = the output column (what we want the model to predict)
# This is called the "target" or "label"
y = df["final_score"]

print(f"      ✅ Input features (X): {list(X.columns)}")
print(f"      ✅ Target (y): final_score")

# -------------------------------------------------------
# STEP 3: Split into Training and Testing sets
# -------------------------------------------------------
print("\n[3/5] Splitting data into train/test sets...")

# We use 80% of data to TRAIN the model, and 20% to TEST it
# random_state=42 means the split will be the same every time (reproducible)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% goes to testing
    random_state=42     # Makes the split reproducible
)

print(f"      ✅ Training samples: {len(X_train)}")
print(f"      ✅ Testing samples:  {len(X_test)}")

# -------------------------------------------------------
# STEP 4: Scale the features (normalize numbers)
# -------------------------------------------------------
# StandardScaler makes all numbers have a similar range.
# Example: attendance (0-100) and study_hours (0-10) have different scales.
# Scaling fixes this so one feature doesn't dominate the others.

scaler = StandardScaler()

# fit_transform: learn the scale from training data, then apply it
X_train_scaled = scaler.fit_transform(X_train)

# transform: apply the SAME scale to test data (don't re-learn from test!)
X_test_scaled = scaler.transform(X_test)

print("\n[4/5] Training the Random Forest model...")

# -------------------------------------------------------
# STEP 5: Train the model
# -------------------------------------------------------
# RandomForestRegressor = creates 100 decision trees and averages their answers
# It's great for beginners because it works well without much tuning
model = RandomForestRegressor(
    n_estimators=100,    # Use 100 decision trees
    random_state=42      # Reproducible results
)

# .fit() is where the actual learning happens
model.fit(X_train_scaled, y_train)

print("      ✅ Model trained successfully!")

# -------------------------------------------------------
# STEP 6: Test the model's accuracy
# -------------------------------------------------------
y_predictions = model.predict(X_test_scaled)

# Mean Absolute Error: on average, how many points off are we?
mae = mean_absolute_error(y_test, y_predictions)

# R² score: 1.0 is perfect, 0.0 is random guessing
r2 = r2_score(y_test, y_predictions)

print(f"\n      📊 Model Accuracy:")
print(f"         Mean Absolute Error: {mae:.2f} points")
print(f"         R² Score: {r2:.3f} (1.0 = perfect)")

# -------------------------------------------------------
# STEP 7: Save the model and scaler to files
# -------------------------------------------------------
print("\n[5/5] Saving model and scaler to 'models/' folder...")

# Create the models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the trained model
joblib.dump(model, "models/performance_model.pkl")

# Save the scaler too! We need to use the SAME scaler when predicting
joblib.dump(scaler, "models/scaler.pkl")

print("      ✅ Saved: models/performance_model.pkl")
print("      ✅ Saved: models/scaler.pkl")

print("\n" + "=" * 50)
print("  TRAINING COMPLETE! Ready for Flask.")
print("=" * 50)
print("\nNext step: Run  python app.py  to start the server!\n")