from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class LCARequest(BaseModel):
    """Defines the structure for the input data sent to the /api/lca endpoint."""
    metal: str = Field(..., example="aluminium")
    route: str = Field(..., example="primary")
    country: Optional[str] = Field("IN", example="IN")
    production_tonnes: Optional[float] = Field(1.0, example=100.0)
    energy_mix_fossil: Optional[float] = Field(None, example=0.65)
    recycling_rate: Optional[float] = Field(None, example=0.25)
    transport_km: Optional[float] = Field(None, example=500.0)
    scrap_return_tonnes: Optional[float] = Field(None, example=20.0)

class FlowItem(BaseModel):
    """Represents a single flow in the Sankey diagram."""
    source: str
    target: str
    value: float = Field(...)

class LCAResponse(BaseModel):
    """Defines the structure of the JSON response from the /api/lca endpoint."""
    inputs: Dict[str, Any]
    environmental_impacts: Dict[str, float]
    circularity_metrics: Dict[str, float]
    material_flows: List[FlowItem]
    recommendations: List[str]