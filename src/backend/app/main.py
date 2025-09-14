from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .api_models import LCARequest, LCAResponse
from .model_server import ModelServer
from .lca_engine import compute_lca_from_inputs

# Initialize FastAPI app
app = FastAPI(
    title="AI-Powered LCA API for Metallurgy",
    description="API to perform Life Cycle Assessments with AI-powered parameter estimation.",
    version="1.0.0"
)

# Allow CORS for the Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the ML model at startup
try:
    model_server = ModelServer(model_path="models/lca_model.pkl")
    print("✅ ML model loaded successfully from Pune.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model_server = None

@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint to confirm API is running."""
    return {"status": "ok", "model_loaded": model_server is not None and model_server.model is not None}

@app.post("/api/lca", response_model=LCAResponse, tags=["LCA"])
def run_lca_analysis(request: LCARequest):
    """Main endpoint to run the Life Cycle Assessment."""
    if not model_server or not model_server.model:
        raise HTTPException(status_code=503, detail="ML Model not available.")
        
    try:
        # 1. Use AI to estimate missing parameters
        estimated_inputs = model_server.estimate_missing_parameters(request.dict())
        # 2. Compute final LCA metrics
        lca_result = compute_lca_from_inputs(estimated_inputs)
        return lca_result
    except Exception as e:
        print(f"Error during LCA computation: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")