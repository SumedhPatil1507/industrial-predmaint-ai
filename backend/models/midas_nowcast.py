"""Mixed-frequency MIDAS nowcasting for monthly CPI inflation.

The model regresses monthly year-over-year CPI inflation on its one-month lag
and exponentially Almon-weighted daily yield observations. A single weight
curve is shared across yield tenors. Inputs are explicitly cutoff by date so a
nowcast cannot accidentally consume observations published after its as-of.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


@dataclass(frozen=True)
class NowcastResult:
    target_month: str
    as_of: str
    cpi_yoy_pct: float
    lower_residual_band_pct: float
    upper_residual_band_pct: float
    daily_observations: int
    training_months: int
    yield_features: Mapping[str, float]


def _monthly_cpi_series(cpi: pd.Series) -> pd.Series:
    if not isinstance(cpi, pd.Series):
        raise TypeError("monthly_cpi must be a pandas Series of CPI index levels")
    result = cpi.copy()
    result.index = pd.PeriodIndex(result.index, freq="M")
    result = pd.to_numeric(result, errors="coerce").groupby(level=0).last().sort_index()
    result = result[np.isfinite(result) & (result > 0)]
    if result.index.has_duplicates:
        result = result.groupby(level=0).last()
    result = result.reindex(pd.period_range(result.index.min(), result.index.max(), freq="M"))
    if len(result) < 13:
        raise ValueError("at least 13 monthly CPI index levels are required for YoY inflation")
    return result


def _yield_frame(yields: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(yields, pd.Series):
        frame = yields.to_frame(name=yields.name or "yield")
    elif isinstance(yields, pd.DataFrame) and not yields.empty:
        frame = yields.copy()
    else:
        raise TypeError("daily_yields must be a non-empty pandas Series or DataFrame")
    if frame.empty or not all(isinstance(name, str) and name for name in frame.columns):
        raise ValueError("daily yield columns must have non-empty string names")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True)).tz_convert(None).normalize()
    frame = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    frame = frame.groupby(level=0).last().sort_index()
    if frame.dropna(how="all").empty:
        raise ValueError("daily_yields contains no finite observations")
    return frame


def _almon_average(frame: pd.DataFrame, theta: np.ndarray) -> np.ndarray:
    if frame.empty:
        return np.full(frame.shape[1], np.nan, dtype=float)
    day = frame.index.day.to_numpy(dtype=float)
    days_in_month = frame.index.days_in_month.to_numpy(dtype=float)
    position = (day - 1.0) / np.maximum(days_in_month - 1.0, 1.0)
    exponent = np.clip(theta[0] * position + theta[1] * position**2, -40.0, 40.0)
    weights = np.exp(exponent)
    values = frame.to_numpy(dtype=float)
    output = np.empty(values.shape[1], dtype=float)
    for column in range(values.shape[1]):
        valid = np.isfinite(values[:, column])
        if not valid.any():
            output[column] = np.nan
            continue
        local_weights = weights[valid]
        output[column] = float(np.dot(local_weights, values[valid, column]) / local_weights.sum())
    return output


class MidasNowcaster:
    """Fit and run a daily-yields to monthly-CPI exponential-Almon MIDAS model."""

    def __init__(self, min_daily_observations: int = 10, confidence_z: float = 1.96) -> None:
        if min_daily_observations < 1:
            raise ValueError("min_daily_observations must be positive")
        if not np.isfinite(confidence_z) or confidence_z <= 0:
            raise ValueError("confidence_z must be finite and positive")
        self.min_daily_observations = min_daily_observations
        self.confidence_z = confidence_z
        self._fitted = False

    def fit(self, monthly_cpi: pd.Series, daily_yields: pd.Series | pd.DataFrame) -> "MidasNowcaster":
        """Fit on CPI index levels and same-calendar-month daily yield histories."""
        cpi = _monthly_cpi_series(monthly_cpi)
        yields = _yield_frame(daily_yields)
        inflation = cpi.pct_change(12).mul(100.0)
        records: list[tuple[pd.Period, float, float]] = []
        for month in inflation.index:
            previous_month = month - 1
            if previous_month not in inflation.index:
                continue
            daily = yields.loc[yields.index.to_period("M") == month]
            if len(daily.dropna(how="all")) < self.min_daily_observations:
                continue
            weighted = _almon_average(daily, np.zeros(2))
            if np.isfinite(inflation.loc[month]) and np.isfinite(inflation.loc[previous_month]) and np.isfinite(weighted).all():
                records.append((month, float(inflation.loc[month]), float(inflation.loc[previous_month])))

        required = max(24, yields.shape[1] + 8)
        if len(records) < required:
            raise ValueError(f"MIDAS fit needs at least {required} complete CPI/yield months; got {len(records)}")
        months = [row[0] for row in records]
        y = np.asarray([row[1] for row in records], dtype=float)
        lagged_cpi = np.asarray([row[2] for row in records], dtype=float)
        feature_count = yields.shape[1]
        x = np.vstack([
            _almon_average(yields.loc[yields.index.to_period("M") == month], np.zeros(2))
            for month in months
        ])

        def residual(parameters: np.ndarray) -> np.ndarray:
            coefficients = parameters[: 2 + feature_count]
            theta = parameters[-2:]
            weighted_x = np.vstack([
                _almon_average(yields.loc[yields.index.to_period("M") == month], theta)
                for month in months
            ])
            predictions = coefficients[0] + coefficients[1] * lagged_cpi + weighted_x @ coefficients[2:]
            return predictions - y

        initial = np.zeros(2 + feature_count + 2, dtype=float)
        initial_design = np.column_stack([np.ones(len(y)), lagged_cpi, x])
        initial[:-2] = np.linalg.lstsq(initial_design, y, rcond=None)[0]
        bounds_low = np.full_like(initial, -100.0)
        bounds_high = np.full_like(initial, 100.0)
        bounds_low[-2:] = -10.0
        bounds_high[-2:] = 10.0
        fit = least_squares(residual, initial, bounds=(bounds_low, bounds_high), loss="soft_l1")
        if not fit.success or not np.isfinite(fit.x).all():
            raise RuntimeError(f"MIDAS optimization failed: {fit.message}")
        errors = residual(fit.x)
        dof = max(1, len(y) - len(fit.x))
        rmse = float(np.sqrt(np.dot(errors, errors) / dof))

        self._coefficients = fit.x[:-2]
        self._theta = fit.x[-2:]
        self._rmse = rmse
        self._yield_columns = list(yields.columns)
        self._latest_cpi_period = cpi.index[-1]
        self._latest_cpi_yoy = float(inflation.iloc[-1])
        if not np.isfinite(self._latest_cpi_yoy):
            raise ValueError("latest CPI month has no available year-over-year rate")
        self._training_months = len(y)
        self._fitted = True
        return self

    def nowcast(
        self,
        daily_yields: pd.Series | pd.DataFrame,
        *,
        target_month: str | date | pd.Period,
        as_of: str | datetime | pd.Timestamp,
    ) -> NowcastResult:
        """Nowcast exactly one month after the last CPI observation, cutoff at ``as_of``."""
        if not self._fitted:
            raise RuntimeError("fit() must be called before nowcast()")
        month = pd.Period(target_month, freq="M")
        if month != self._latest_cpi_period + 1:
            raise ValueError("target_month must immediately follow the latest CPI month used in fit")
        cutoff = pd.Timestamp(as_of)
        if cutoff.tzinfo is None:
            raise ValueError("as_of must include a timezone")
        cutoff = cutoff.tz_convert("UTC").tz_localize(None)
        yields = _yield_frame(daily_yields)
        if list(yields.columns) != self._yield_columns:
            raise ValueError(f"daily yield columns must match fitted columns: {self._yield_columns}")
        month_start = month.start_time
        month_end = (month + 1).start_time
        observed = yields.loc[(yields.index >= month_start) & (yields.index < month_end) & (yields.index <= cutoff)]
        count = len(observed.dropna(how="all"))
        if count < self.min_daily_observations:
            raise ValueError(
                f"nowcast requires {self.min_daily_observations} daily yield observations by as_of; got {count}"
            )
        features = _almon_average(observed, self._theta)
        if not np.isfinite(features).all():
            raise ValueError("each fitted yield series needs a finite observation by as_of")
        coefficients = self._coefficients
        prediction = float(coefficients[0] + coefficients[1] * self._latest_cpi_yoy + features @ coefficients[2:])
        margin = self.confidence_z * self._rmse
        return NowcastResult(
            target_month=str(month),
            as_of=cutoff.replace(tzinfo=timezone.utc).isoformat(),
            cpi_yoy_pct=prediction,
            lower_residual_band_pct=prediction - margin,
            upper_residual_band_pct=prediction + margin,
            daily_observations=count,
            training_months=self._training_months,
            yield_features={name: float(value) for name, value in zip(self._yield_columns, features)},
        )
