"""Point-in-time correctness checks for the industrial Feast feature schema."""
from datetime import timedelta

import pandas as pd
import pytest

feast = pytest.importorskip("feast")

from feast import Entity, FeatureStore, FeatureView, Field, FileSource
from feast.types import Float32, String

from feature_definitions import industrial_sensor_features


def test_industrial_feature_view_has_asset_and_plant_entities():
    assert {entity.name for entity in industrial_sensor_features.entities} == {"asset_id", "plant_id"}
    assert "vibration_rms" in {field.name for field in industrial_sensor_features.schema}
    assert "acoustic_crest_factor" in {field.name for field in industrial_sensor_features.schema}


def test_historical_features_are_point_in_time_correct(tmp_path):
    """A row arriving after the entity timestamp must never appear in its join."""
    source_path = tmp_path / "sensor_features.parquet"
    (tmp_path / "feature_store.yaml").write_text(
        "project: predmaint_test\nregistry: registry.db\nprovider: local\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "asset_id": ["pump-17", "pump-17"],
            "plant_id": ["plant-a", "plant-a"],
            "event_timestamp": pd.to_datetime(
                ["2026-01-10T10:00:00Z", "2026-01-10T10:10:00Z"], utc=True
            ),
            "vibration_rms": [1.25, 99.0],
        }
    ).to_parquet(source_path, index=False)

    asset = Entity(name="asset_id", join_keys=["asset_id"], value_type=String)
    plant = Entity(name="plant_id", join_keys=["plant_id"], value_type=String)
    source = FileSource(
        name="pit_sensor_source",
        path=str(source_path),
        timestamp_field="event_timestamp",
    )
    view = FeatureView(
        name="pit_sensor_features",
        entities=[asset, plant],
        ttl=timedelta(days=1),
        schema=[Field(name="vibration_rms", dtype=Float32)],
        source=source,
    )
    store = FeatureStore(repo_path=str(tmp_path))
    store.apply([asset, plant, view])

    entity_df = pd.DataFrame(
        {
            "asset_id": ["pump-17"],
            "plant_id": ["plant-a"],
            "event_timestamp": pd.to_datetime(["2026-01-10T10:05:00Z"], utc=True),
        }
    )
    joined = store.get_historical_features(
        entity_df=entity_df,
        features=["pit_sensor_features:vibration_rms"],
    ).to_df()

    assert joined.loc[0, "vibration_rms"] == pytest.approx(1.25)
    assert joined.loc[0, "vibration_rms"] != pytest.approx(99.0)
