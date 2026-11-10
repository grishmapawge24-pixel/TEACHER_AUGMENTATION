# =============================================================
# predictor.py
# PURPOSE: Load the saved ML model and provide a function
#          that Flask can call to make predictions.
#
# Think of this file as the "bridge" between Flask and the model.
# =============================================================

import joblib   # For loading our saved model files
import pandas as pd

# --- Load the model and scaler when this file is first imported ---
# This runs ONCE when Flask starts up (not on every request)
# It's efficient because loading from disk takes a tiny bit of time

print("[Predictor] Loading model and scaler from disk...")

# Load the trained Random Forest model
model = joblib.load("models/performance_model.pkl")

# Load the scaler (must use the SAME scaler we trained with!)
scaler = joblib.load("models/scaler.pkl")

print("[Predictor] ✅ Model and scaler loaded successfully!")


def get_risk_level(score):
    """
    Converts a predicted score into a risk category.
    
    This is simple rule-based logic (no ML needed here).
    
    Args:
        score (float): The predicted final score (0-100)
    
    Returns:
        dict: {"level": "High/Medium/Low", "color": "red/orange/green", "message": "..."}
    """
    if score < 50:
        return {
            "level": "High Risk",
            "color": "danger",       # Bootstrap class (red)
            "emoji": "🔴",
            "message": "Immediate intervention required. Student is at serious risk of failing."
        }
    elif score < 70:
        return {
            "level": "Medium Risk",
            "color": "warning",      # Bootstrap class (orange/yellow)
            "emoji": "🟡",
            "message": "Student needs attention. Consider extra support sessions."
        }
    else:
        return {
            "level": "Low Risk",
            "color": "success",      # Bootstrap class (green)
            "emoji": "🟢",
            "message": "Student is performing well. Keep monitoring progress."
        }


def predict_performance(attendance, study_hours, previous_marks, assignments_completed):
    """
    Main prediction function. Takes student data and returns predicted score + risk.
    
    Flask will call this function when a user submits the form.
    
    Args:
        attendance (float): Attendance percentage (0-100)
        study_hours (float): Hours studied per day
        previous_marks (float): Marks in previous exam (0-100)
        assignments_completed (int): Number of assignments done (0-10)
    
    Returns:
        dict: {
            "predicted_score": 75.3,
            "risk": {"level": "Low Risk", "color": "success", ...},
            "input_summary": {...}
        }
    """
    
    # STEP 1: Put the inputs into a 2D array (sklearn expects this format)
    # [[value1, value2, value3, value4]] — note the double brackets!
    input_data = pd.DataFrame(
        [[
            float(attendance),
            float(study_hours),
            float(previous_marks),
            float(assignments_completed),
        ]],
        columns=[
            "attendance",
            "study_hours",
            "previous_marks",
            "assignments_completed",
        ],
    )
    
    # STEP 2: Scale the input using the SAME scaler from training
    input_scaled = scaler.transform(input_data)
    
    # STEP 3: Make the prediction
    # model.predict() returns an array, so we take the first element [0]
    predicted_score = model.predict(input_scaled)[0]
    
    # STEP 4: Clamp the score to 0-100 range (safety check)
    predicted_score = max(0, min(100, predicted_score))
    
    # STEP 5: Get the risk level based on the predicted score
    risk_info = get_risk_level(predicted_score)
    
    # STEP 6: Return everything as a dictionary
    return {
        "predicted_score": round(predicted_score, 1),  # Round to 1 decimal
        "risk": risk_info,
        "input_summary": {
            "attendance": attendance,
            "study_hours": study_hours,
            "previous_marks": previous_marks,
            "assignments_completed": assignments_completed
        }
    }
