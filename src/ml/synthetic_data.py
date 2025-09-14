import pandas as pd
import numpy as np
from pathlib import Path

def generate_synthetic_lca(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic dataset for Life Cycle Assessment (LCA) modeling.
    This function creates a DataFrame simulating LCA data for various metals,
    production routes, and countries, introducing missing values to mimic real-world gaps.
    """
    np.random.seed(seed)
    
    metals = np.random.choice(['aluminium', 'copper', 'nickel', 'lithium'], size=n_samples, p=[0.4, 0.4, 0.1, 0.1])
    routes = np.random.choice(['primary', 'secondary', 'mixed'], size=n_samples, p=[0.5, 0.3, 0.2])
    countries = np.random.choice(['IN', 'CN', 'US', 'AU', 'RU', 'BR'], size=n_samples)
    
    energy_mix_fossil = np.random.uniform(0.2, 1.0, n_samples)
    production_tonnes = np.random.exponential(scale=2.0, size=n_samples) + 0.1
    transport_km = np.random.exponential(scale=500, size=n_samples)
    
    recycling_rate = np.where(
        routes == 'secondary', 
        np.random.uniform(0.6, 0.95, n_samples), 
        np.random.uniform(0.05, 0.45, n_samples)
    )
    scrap_return_tonnes = recycling_rate * production_tonnes * np.random.uniform(0.4, 1.0, n_samples)

    base_energy = np.where(metals == 'aluminium', 150000, 50000)
    route_multiplier = np.where(routes == 'primary', 1.0, 0.15)
    fossil_fuel_penalty = 1 + (energy_mix_fossil * 0.4)
    noise = np.random.normal(1, 0.05, n_samples)
    energy_intensity = base_energy * route_multiplier * fossil_fuel_penalty * noise
    ghg_kgCO2e = energy_intensity * (0.025 * energy_mix_fossil) + np.random.normal(0, 50, n_samples)

    df = pd.DataFrame({
        'metal': metals, 'route': routes, 'country': countries,
        'energy_mix_fossil': energy_mix_fossil, 'production_tonnes': production_tonnes,
        'recycling_rate': recycling_rate, 'transport_km': transport_km,
        'scrap_return_tonnes': scrap_return_tonnes, 'energy_intensity_MJ_per_tonne': energy_intensity,
        'ghg_kgCO2e_per_tonne': ghg_kgCO2e
    })

    # Introduce missingness for the model to predict
    for col in ['energy_mix_fossil', 'recycling_rate', 'transport_km', 'energy_intensity_MJ_per_tonne']:
        mask = np.random.rand(n_samples) < 0.15
        df.loc[mask, col] = np.nan

    return df

if __name__ == "__main__":
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    output_path = data_dir / "sample_synthetic.csv"
    synthetic_df = generate_synthetic_lca(2000)
    synthetic_df.to_csv(output_path, index=False)
    print(f"✅ Synthetic data generated and saved to: {output_path}")