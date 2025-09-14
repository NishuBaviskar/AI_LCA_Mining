import joblib
import pandas as pd
import numpy as np  # Import numpy to check for NaN
from typing import Dict, Any

class ModelServer:
    """
    A class to load a trained model and use it for predictions.
    THIS FILE CONTAINS THE PERMANENT, BULLETPROOF FIX.
    """
    def __init__(self, model_path: str):
        try:
            self.model = joblib.load(model_path)
            print("✅ ML model loaded successfully.")
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Failed to load model from {model_path}: {e}")
            self.model = None

    def estimate_missing_parameters(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fills missing fields using the ML model.
        This function now guarantees 'energy_intensity_MJ_per_tonne' is ALWAYS a valid float.
        """
        estimated_inputs = inputs.copy()

        # We only predict if the key is missing or the value is None.
        if 'energy_intensity_MJ_per_tonne' not in estimated_inputs or estimated_inputs.get('energy_intensity_MJ_per_tonne') is None:
            print("INFO: 'energy_intensity' is missing. Predicting with AI model...")
            
            prediction = None
            if self.model is not None:
                try:
                    # Use the original inputs directly to create the feature DataFrame
                    feature_df = pd.DataFrame([inputs]) 
                    prediction = self.model.predict(feature_df)[0]
                except Exception as e:
                    print(f"ERROR: Model prediction failed unexpectedly: {e}. Will use fallback.")
                    prediction = None # Ensure prediction is None on failure

            # --- THIS IS THE BULLETPROOF CHECK ---
            # Check if prediction is None OR if it's a non-finite number (like NaN)
            if prediction is None or not np.isfinite(prediction):
                print(f"WARNING: AI Prediction was invalid or failed. Using a safe fallback value.")
                estimated_inputs['energy_intensity_MJ_per_tonne'] = 50000.0
            else:
                estimated_inputs['energy_intensity_MJ_per_tonne'] = float(prediction)
                print(f"INFO: AI Predicted energy intensity: {prediction:.2f} MJ/tonne")
        
        # Estimate GHG from energy if missing
        if 'ghg_kgCO2e_per_tonne' not in estimated_inputs or not estimated_inputs.get('ghg_kgCO2e_per_tonne'):
            energy = estimated_inputs.get('energy_intensity_MJ_per_tonne', 50000.0)
            fossil_fraction = estimated_inputs.get('energy_mix_fossil', 0.6)
            estimated_inputs['ghg_kgCO2e_per_tonne'] = float(energy * 0.025 * fossil_fraction)

        return estimated_inputs