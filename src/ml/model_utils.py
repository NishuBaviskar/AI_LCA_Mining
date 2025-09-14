import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor

def build_feature_pipeline() -> ColumnTransformer:
    """Builds a scikit-learn pipeline for preprocessing features."""
    cat_cols = ['metal', 'route', 'country']
    num_cols = ['energy_mix_fossil', 'production_tonnes', 'recycling_rate', 'transport_km', 'scrap_return_tonnes']

    numeric_transformer = SimpleImputer(strategy='median')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ],
        remainder='passthrough'
    )
    return preprocessor

def train_lca_model(df: pd.DataFrame) -> Pipeline:
    """Trains an XGBoost model to predict energy intensity."""
    df_train = df.dropna(subset=['energy_intensity_MJ_per_tonne']).copy()
    
    features = ['metal', 'route', 'country', 'energy_mix_fossil', 'production_tonnes', 'recycling_rate', 'transport_km', 'scrap_return_tonnes']
    target = 'energy_intensity_MJ_per_tonne'
    
    X = df_train[features]
    y = df_train[target]
    
    preprocessor = build_feature_pipeline()
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42, objective='reg:squarederror'))
    ])
    
    print("🚀 Starting model training...")
    model_pipeline.fit(X, y)
    print("✅ Model training complete.")
    
    return model_pipeline