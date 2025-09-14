import pandas as pd
import argparse
import joblib
from pathlib import Path
from model_utils import train_lca_model
from synthetic_data import generate_synthetic_lca

def main():
    """Main function to run the model training pipeline."""
    parser = argparse.ArgumentParser(description="Train the AI LCA model.")
    parser.add_argument("--data", type=str, default="data/sample_synthetic.csv", help="Path to the training data CSV file.")
    parser.add_argument("--out", type=str, default="models/lca_model.pkl", help="Path to save the trained model (.pkl file).")
    args = parser.parse_args()

    data_path = Path(args.data)
    try:
        df = pd.read_csv(data_path)
        print(f"📖 Loaded training data from '{data_path}'")
    except FileNotFoundError:
        print(f"⚠️ Data file not found at '{data_path}'. Generating synthetic data.")
        df = generate_synthetic_lca(2000)
        data_path.parent.mkdir(exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"💾 Saved new synthetic data to '{data_path}'")

    lca_model_pipeline = train_lca_model(df)

    model_path = Path(args.out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(lca_model_pipeline, model_path)
    print(f"✅ Model successfully trained and saved to '{model_path}'")

if __name__ == "__main__":
    main()