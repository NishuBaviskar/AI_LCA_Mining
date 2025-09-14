import requests
from typing import Optional, Dict, Any

# World Bank API configuration for CO2 emissions from electricity production
BASE_URL = "http://api.worldbank.org/v2/country"
EMISSIONS_INDICATOR = "EG.ELC.CO2E.KH" # kg per kWh
FORMAT = "json"

def fetch_country_emissions_data(country_code: str) -> Optional[Dict[str, Any]]:
    """
    Fetches the most recent CO2 emissions data for a country from the World Bank API.
    This provides a real-world factor for grid carbon intensity.
    """
    api_url = f"{BASE_URL}/{country_code}/indicator/{EMISSIONS_INDICATOR}?format={FORMAT}&per_page=1"
    
    print(f"INFO: Calling World Bank API: {api_url}")
    
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Parse the JSON response, which has a specific structure
        if len(data) > 1 and data[1]:
            latest_data_point = data[1][0]
            value = latest_data_point.get('value')
            if value is not None:
                return {
                    "country": latest_data_point['country']['value'],
                    "year": latest_data_point['date'],
                    "kg_co2_per_kwh": float(value),
                    "source": "The World Bank"
                }
        return None
    except requests.exceptions.RequestException as e:
        print(f"ERROR: World Bank API connection failed: {e}")
        return None
    except Exception as e:
        print(f"ERROR: Failed to parse World Bank API response: {e}")
        return None