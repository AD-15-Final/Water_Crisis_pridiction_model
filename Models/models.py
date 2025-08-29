from tensorflow.keras.models import load_model

# --- Load the Saved Model ---
loaded_model = load_model("chennai_bilstm_artifacts\optimised_bilstm.keras")
print("Model loaded successfully.")

# --- Prepare input for prediction ---
last_days_scaled = X_scaled[-time_step:]   # last 90 days (since time_step=90)
prediction_input = last_days_scaled.reshape(1, time_step, X_df.shape[1])

# --- Make Prediction ---
final_pred_probs = loaded_model.predict(prediction_input)
predicted_class = np.argmax(final_pred_probs, axis=1)[0]

# --- Display the Prediction ---
last_date_in_data = X_df.index[-1]
prediction_date = last_date_in_data + pd.Timedelta(days=1)

print("\n\n=====================================================")
print(f"      OPTIMIZED PREDICTION FOR CHENNAI WATER CRISIS")
print("=====================================================")
print(f"Based on data up to: {last_date_in_data.strftime('%Y-%m-%d')}")
print(f"Predicted Crisis Level for {prediction_date.strftime('%Y-%m-%d')}: {predicted_class}")
print("=====================================================")
print("\nCrisis Levels:")
print("0: No Crisis")
print("1: Moderate Crisis")
print("2: Severe Crisis")
print("3: Extreme Crisis")
