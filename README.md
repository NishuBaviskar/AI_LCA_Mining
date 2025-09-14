# AI-Driven LCA Tool for Metallurgy & Mining

**Project for Smart India Hackathon 2025: AI for Sustainability**

This project is an AI-powered Life Cycle Assessment (LCA) tool designed to promote circular economy principles in the metallurgy and mining sectors.

**New in this version:** The tool now integrates **real-time data** from the World Bank API to provide more accurate, country-specific emissions calculations.



## 🌟 Highlights

-   **Real-Time Data Enrichment:** Dynamically fetches country-specific CO2 emissions factors from the World Bank API, making LCA results more accurate and relevant.
-   **AI-Powered Estimation:** An XGBoost model intelligently predicts missing LCA parameters (e.g., energy intensity) when primary data is unavailable.
-   **Interactive Visualizations:** A Streamlit dashboard visualizes environmental impacts and material flows with Plotly-based Sankey diagrams.
-   **Circularity Focus:** Calculates a circularity index and provides actionable recommendations to improve sustainability.

## 🚀 Quick Start

1.  **Create a virtual environment and install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Generate Synthetic Data (for model training):**
    ```bash
    python src/ml/synthetic_data.py
    ```

3.  **Train the ML Model:**
    ```bash
    python src/ml/train.py --data data/sample_synthetic.csv --out models/lca_model.pkl
    ```

4.  **Run the Backend Server (with live data connection):**
    ```bash
    uvicorn src.backend.app.main:app --reload --port 8000
    ```

5.  **Run the Frontend Application (in a new terminal):**
    ```bash
    python -m streamlit run src/frontend/streamlit_app.py
    ```

6.  **Access the Tool:**
    -   **Frontend UI:** `http://localhost:8501`
    -   **Backend API Docs:** `http://localhost:8000/docs`"# AI_LCA_Mining" 
