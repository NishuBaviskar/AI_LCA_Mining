import streamlit as st
import requests
import json
import plotly.graph_objects as go
import os
from dotenv import load_dotenv
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# --- PAGE CONFIG & SETUP ---
st.set_page_config(page_title="AI-LCA Tool for Metallurgy", page_icon="♻️", layout="wide")
load_dotenv()

# --- CONFIGURATION ---
# The API_URL is no longer needed as the backend logic is now part of this file.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- CSS for Floating Action Button (FAB) ---
# This is the exact CSS from your working file. No changes made.
st.markdown("""
<style>
    div[data-testid="stButton-fab"] button {
        position: fixed;
        bottom: 30px;
        right: 30px;
        height: 60px;
        width: 60px;
        border-radius: 50%;
        border: none;
        background: linear-gradient(135deg, #2563EB, #1D4ED8); /* Modern blue gradient */
        color: white;
        font-size: 26px;
        font-weight: bold;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        z-index: 1000;
        cursor: pointer;
        transition: all 0.25s ease-in-out;
    }

    div[data-testid="stButton-fab"] button:hover {
        background: linear-gradient(135deg, #1E40AF, #1D4ED8);
        transform: scale(1.08);
        box-shadow: 0 6px 16px rgba(0,0,0,0.35);
    }

    div[data-testid="stButton-fab"] button:active {
        transform: scale(0.95);
        background: linear-gradient(135deg, #1E3A8A, #1D4ED8);
    }
</style>
""", unsafe_allow_html=True)


# --- BACKEND LOGIC (Integrated directly into the Streamlit App) ---

# This is the official Streamlit way to load a large object like a model only once.
@st.cache_resource
def load_model():
    """Loads the trained ML model from disk, caching it for performance."""
    model_path = Path("models/lca_model.pkl")
    if model_path.exists():
        try:
            model = joblib.load(model_path)
            print("✅ ML model loaded successfully.")
            return model
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Failed to load model: {e}")
            return None
    else:
        st.error("Model file (models/lca_model.pkl) not found! Please ensure the file exists.")
        return None

def estimate_missing_parameters(_model, inputs: dict):
    """Fills missing fields using the loaded ML model."""
    if 'energy_intensity_MJ_per_tonne' not in inputs or inputs.get('energy_intensity_MJ_per_tonne') is None:
        if _model:
            try:
                feature_df = pd.DataFrame([inputs])
                prediction = _model.predict(feature_df)[0]
                if np.isfinite(prediction):
                    inputs['energy_intensity_MJ_per_tonne'] = float(prediction)
                else:
                    inputs['energy_intensity_MJ_per_tonne'] = 50000.0
            except Exception:
                inputs['energy_intensity_MJ_per_tonne'] = 50000.0
        else:
            inputs['energy_intensity_MJ_per_tonne'] = 50000.0
    return inputs

def fetch_country_emissions_data(country_code: str):
    """Fetches real-time CO2 emissions data for a country from the World Bank API."""
    BASE_URL = "http://api.worldbank.org/v2/country"
    EMISSIONS_INDICATOR = "EG.ELC.CO2E.KH"
    api_url = f"{BASE_URL}/{country_code}/indicator/{EMISSIONS_INDICATOR}?format=json&per_page=1"
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if len(data) > 1 and data[1]:
            latest = data[1][0]
            if latest.get('value') is not None:
                return {"year": latest['date'], "kg_co2_per_kwh": float(latest['value'])}
        return None
    except requests.exceptions.RequestException:
        return None

def compute_lca_from_inputs(inputs: dict):
    """Computes final LCA metrics from a complete set of inputs."""
    prod_tonnes = inputs.get('production_tonnes') or 1.0
    energy_per_tonne_mj = inputs.get('energy_intensity_MJ_per_tonne') or 50000.0
    recycling_rate = inputs.get('recycling_rate') or 0.0
    scrap_return = inputs.get('scrap_return_tonnes') or (recycling_rate * prod_tonnes)
    transport_km = inputs.get('transport_km') or 0.0
    ghg_per_tonne = inputs.get('ghg_kgCO2e_per_tonne') or 0.0
    country_code = inputs.get('country', 'IN').upper()

    real_time_source = "AI Model Fallback Estimate"
    live_emissions_data = fetch_country_emissions_data(country_code)
    if live_emissions_data:
        energy_kwh = energy_per_tonne_mj / 3.6
        ghg_per_tonne = energy_kwh * live_emissions_data['kg_co2_per_kwh']
        real_time_source = f"The World Bank ({live_emissions_data['year']})"

    inputs['ghg_kgCO2e_per_tonne'] = ghg_per_tonne
    inputs['ghg_data_source'] = real_time_source

    total_ghg = (ghg_per_tonne * prod_tonnes) + (0.15 * transport_km * prod_tonnes)
    impacts = {'total_ghg_kgCO2e': round(total_ghg, 2), 'total_energy_MJ': round(energy_per_tonne_mj * prod_tonnes, 2)}
    
    circularity = {'circularity_index': round((0.5 * recycling_rate) + (0.3 * (scrap_return / (prod_tonnes + 1e-6))) + (0.2 * (1 if inputs.get('route') == 'secondary' else 0)), 3),
                   'material_waste_tonnes': round(prod_tonnes * (1 - recycling_rate), 2)}

    primary_input = prod_tonnes if inputs.get('route') != 'secondary' else max(0, prod_tonnes - scrap_return)
    flows = [{'source': 'Primary Feedstock', 'target': 'Processing', 'value': round(primary_input, 2)},
             {'source': 'Recycled Scrap', 'target': 'Processing', 'value': round(scrap_return, 2)},
             {'source': 'Processing', 'target': 'Manufacturing', 'value': round(prod_tonnes, 2)},
             {'source': 'Manufacturing', 'target': 'End of Life', 'value': round(prod_tonnes, 2)},
             {'source': 'End of Life', 'target': 'Recycled Scrap', 'value': round(prod_tonnes * recycling_rate, 2)},
             {'source': 'End of Life', 'target': 'Landfill/Waste', 'value': round(prod_tonnes * (1 - recycling_rate), 2)}]
    flows = [f for f in flows if f['value'] > 0.01]
    
    recs = ["Enhance collection systems to increase recycling.", "Explore local supply chains to reduce transport emissions.", "Investigate sourcing secondary (recycled) feedstock.", "Transition towards renewable energy sources for processing."]

    return {'inputs': inputs, 'environmental_impacts': impacts, 'circularity_metrics': circularity, 'material_flows': flows, 'recommendations': recs}


# --- UI HELPER FUNCTIONS ---
def get_gemini_response(user_query: str):
    if not GEMINI_API_KEY: 
        return "Error: GEMINI_API_KEY is not configured in your .env file."
    system_prompt = "You are an expert AI assistant for the 'AI-Powered LCA Tool for Metallurgy'. Help users with errors or questions about the app's features."
    gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": f"{system_prompt}\n\nUser Question: {user_query}"}]}]}
    try:
        response = requests.post(gemini_api_url, json=payload, timeout=45)
        response.raise_for_status()
        result = response.json()
        return result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Sorry, I couldn't process that.")
    except Exception as e:
        return f"An error occurred while contacting the AI assistant: {e}"

def create_sankey_chart(flow_data):
    if not flow_data: 
        return go.Figure()
    labels = list(set([f['source'] for f in flow_data] + [f['target'] for f in flow_data]))
    label_indices = {label: i for i, label in enumerate(labels)}
    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=25, thickness=20, label=labels, color="#3B82F6"),
        link=dict(source=[label_indices[f['source']] for f in flow_data],
                  target=[label_indices[f['target']] for f in flow_data],
                  value=[f['value'] for f in flow_data])
    )])
    fig.update_layout(title_text="Material Flow Analysis", font_size=12, height=400)
    return fig

# --- INITIALIZE SESSION STATE ---
if 'lca_results' not in st.session_state: st.session_state.lca_results = None
if "messages" not in st.session_state: st.session_state["messages"] = []
if "show_chat" not in st.session_state: st.session_state.show_chat = False

# --- Button toggle handler ---
def toggle_chat():
    st.session_state.show_chat = not st.session_state.show_chat

# --- MAIN APP UI ---
st.title("♻️ AI-Powered LCA Tool for Metallurgy & Mining")
st.markdown("An interactive tool by team Pune, Maharashtra for Smart India Hackathon 2025.")

# --- Load the model once at the start ---
model = load_model()

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ LCA Input Parameters")
    metal = st.selectbox("Metal", ["aluminium", "copper", "nickel", "lithium"])
    route = st.selectbox("Route", ["primary", "secondary", "mixed"])
    country = st.selectbox("Country", ["IN", "CN", "US", "AU", "DE"])
    production_tonnes = st.number_input("Production Volume (tonnes)", value=100.0)
    st.markdown("---")
    energy_mix_fossil = st.slider("Fossil Fuel in Energy Mix (%)", 0, 100, 60) / 100.0
    recycling_rate = st.slider("End-of-Life Recycling Rate (%)", 0, 100, 25) / 100.0
    transport_km = st.number_input("Avg. Transport Distance (km)", value=500.0)
    
    if st.button("🚀 Run LCA Analysis", type="primary", use_container_width=True):
        if model:
            payload = {"metal": metal, "route": route, "country": country, "production_tonnes": production_tonnes,
                       "energy_mix_fossil": energy_mix_fossil, "recycling_rate": recycling_rate, "transport_km": transport_km}
            
            with st.spinner("🧠 Performing LCA..."):
                # --- THIS IS THE KEY CHANGE: Call local Python functions instead of an API ---
                estimated_inputs = estimate_missing_parameters(model, payload)
                st.session_state.lca_results = compute_lca_from_inputs(estimated_inputs)
        else:
            st.error("Model not loaded. Cannot run analysis. Please ensure 'models/lca_model.pkl' exists.")
        st.rerun()

# --- Display Main Dashboard ---
if st.session_state.get('api_error'):
    st.error(st.session_state.api_error)

if st.session_state.get('lca_results'):
    results = st.session_state.lca_results
    st.success("✅ Analysis Complete!")
    ghg_source = results.get('inputs', {}).get('ghg_data_source', 'AI Model Estimate')
    st.info(f"💡 **GHG Emissions Factor Source:** `{ghg_source}`")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total GHG (kg CO₂e)", f"{results['environmental_impacts']['total_ghg_kgCO2e']:,.0f}")
    col2.metric("Circularity Index", f"{results['circularity_metrics']['circularity_index']:.2f}")
    col3.metric("Material Waste (tonnes)", f"{results['circularity_metrics']['material_waste_tonnes']:,.1f}")
    st.markdown("---")
    left, right = st.columns([2, 1])
    with left:
        st.subheader("📊 Material Flow Sankey Diagram")
        st.plotly_chart(create_sankey_chart(results['material_flows']), use_container_width=True)
    with right:
        st.subheader("💡 Actionable Recommendations")
        for rec in results['recommendations']: 
            st.info(f"- {rec}")
        st.download_button("📥 Download Report (JSON)", json.dumps(results, indent=2),
                           "LCA_Report.json", "application/json", use_container_width=True)
    with st.expander("🔬 View Detailed Data"):
        st.json(results)

# --- STABLE CHATBOT IMPLEMENTATION ---
st.button("🤖", key="fab", help="AI Assistant", on_click=toggle_chat)

if st.session_state.show_chat:
    with st.expander("🤖 AI Assistant", expanded=True):
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input("Ask a question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            with st.spinner("AI is thinking..."):
                response = get_gemini_response(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.chat_message("assistant").write(response)
                st.rerun()
