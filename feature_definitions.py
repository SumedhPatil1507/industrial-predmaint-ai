"""Feast entities and time-aware industrial sensor feature view."""
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, String

asset = Entity(name="asset_id", join_keys=["asset_id"], value_type=String)
plant = Entity(name="plant_id", join_keys=["plant_id"], value_type=String)

sensor_source = FileSource(
    name="industrial_sensor_features_source",
    path="data/feast/industrial_sensor_features.parquet",
    timestamp_field="event_timestamp",
)

industrial_sensor_features = FeatureView(
    name="industrial_sensor_features",
    entities=[asset, plant],
    ttl=timedelta(days=30),
    schema=[
        Field(name=f"{sensor}_{stat}", dtype=Float32)
        for sensor in ("vibration", "temperature", "pressure", "acoustic")
        for stat in ("rms", "kurtosis", "crest_factor")
    ],
    source=sensor_source,
    online=True,
    tags={"domain": "industrial_iot", "purpose": "predictive_maintenance"},
)
