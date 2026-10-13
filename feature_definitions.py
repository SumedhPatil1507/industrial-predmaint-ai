"""Feast entities and time-aware industrial sensor feature view."""
from datetime import timedelta

from feast import Entity, FeatureView, Field
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import PostgreSQLSource
from feast.types import Float64, String

asset = Entity(name="asset_id", join_keys=["asset_id"], value_type=String)
plant = Entity(name="plant_id", join_keys=["plant_id"], value_type=String)

sensor_source = PostgreSQLSource(
    name="industrial_sensor_features_source",
    query="SELECT * FROM sensor_telemetry",
    timestamp_field="event_timestamp",
)

industrial_sensor_features = FeatureView(
    name="industrial_sensor_features",
    entities=[asset, plant],
    ttl=timedelta(days=30),
    schema=[
        Field(name=f"{sensor}_{stat}", dtype=Float64)
        for sensor in ("vibration", "thermal", "acoustic")
        for stat in ("rms", "kurtosis", "crest_factor")
    ],
    source=sensor_source,
    online=True,
    tags={"domain": "industrial_iot", "purpose": "predictive_maintenance"},
)
