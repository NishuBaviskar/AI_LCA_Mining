import json
import pandas as pd
from pathlib import Path
import argparse

def json_to_excel(lca_data: dict, output_path: Path):
    """Converts the LCA JSON result into a multi-sheet Excel report."""
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            pd.json_normalize(lca_data['inputs']).T.rename(columns={0: 'Value'}).to_excel(writer, sheet_name="Inputs & Estimates")
            pd.DataFrame.from_dict(lca_data['environmental_impacts'], orient='index', columns=['Value']).to_excel(writer, sheet_name="Environmental Impacts")
            pd.DataFrame.from_dict(lca_data['circularity_metrics'], orient='index', columns=['Value']).to_excel(writer, sheet_name="Circularity Metrics")
            pd.DataFrame(lca_data['recommendations'], columns=["Actionable Recommendations"]).to_excel(writer, sheet_name="Recommendations", index=False)
        print(f"✅ Excel report successfully saved to {output_path}")
    except Exception as e:
        print(f"❌ Failed to create Excel report: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate reports from LCA JSON output.")
    parser.add_argument("json_file", type=str, help="Path to the input LCA result JSON file.")
    args = parser.parse_args()
    
    input_path = Path(args.json_file)
    if not input_path.exists():
        print(f"❌ Error: Input file not found at '{input_path}'"); return

    output_dir = Path("reports_output"); output_dir.mkdir(exist_ok=True)
    excel_output_path = output_dir / f"{input_path.stem}_report.xlsx"

    with open(input_path, 'r') as f:
        lca_results = json.load(f)
    
    json_to_excel(lca_results, excel_output_path)

if __name__ == "__main__":
    main()