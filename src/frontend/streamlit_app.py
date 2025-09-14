import streamlit as st
import requests
import json
import plotly.graph_objects as go
import os
from dotenv import load_dotenv

# --- PAGE CONFIG & SETUP ---
st.set_page_config(page_title="AI-LCA Tool for Metallurgy", page_icon="♻️", layout="wide")
load_dotenv()

# --- API Configuration ---
API_URL = "http://localhost:8000/api/lca"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- CSS for Floating Action Button (FAB) ---
# Fixed: Now targets the button by its Streamlit key ("fab") instead of title
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


# --- HELPER FUNCTIONS ---
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
if 'api_error' not in st.session_state: st.session_state.api_error = None
if "messages" not in st.session_state: st.session_state["messages"] = []
if "show_chat" not in st.session_state: st.session_state.show_chat = False

# --- NEW: Button toggle handler ---
def toggle_chat():
    st.session_state.show_chat = not st.session_state.show_chat

# --- MAIN APP UI ---
st.title("♻️ AI-Powered LCA Tool for Metallurgy & Mining")
st.markdown("An interactive tool by team Pune, Maharashtra for Smart India Hackathon 2025.")

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
        payload = {"metal": metal, "route": route, "country": country, "production_tonnes": production_tonnes,
                   "energy_mix_fossil": energy_mix_fossil, "recycling_rate": recycling_rate, "transport_km": transport_km}
        with st.spinner("🧠 Performing LCA... Contacting World Bank servers..."):
            try:
                res = requests.post(API_URL, json=payload, timeout=20)
                res.raise_for_status()
                st.session_state.lca_results = res.json()
                st.session_state.api_error = None
            except requests.exceptions.HTTPError as e:
                st.session_state.api_error = f"API Error: Backend server error. Details: {e}"
                st.session_state.lca_results = None
            except requests.exceptions.RequestException as e:
                st.session_state.api_error = f"API Error: Cannot connect to backend. Is it running? Details: {e}"
                st.session_state.lca_results = None
        st.rerun()

# --- Display Main Dashboard ---
if st.session_state.api_error:
    st.error(st.session_state.api_error)

if st.session_state.lca_results:
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
