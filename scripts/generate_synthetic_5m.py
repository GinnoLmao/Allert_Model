"""Generate a realistic 5-month synthetic air-quality dataset for ALLERT.

The output preserves the same 4 variables as the original 3-month dataset:
    timestamp, temperature_c, humidity_percent, pm2_5, pm10

It is generated from the statistics of the original data plus:
    - realistic diurnal temperature cycle
    - temperature-humidity anti-correlation
    - smooth autoregressive PM evolution
    - occasional multi-day pollution episodes
    - PM10/PM2.5 ratio sampled from the observed distribution
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_original_stats(path: Path) -> dict:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    # Use only numeric columns for stats
    stats = {}
    for col in ["temperature_c", "humidity_percent", "pm2_5", "pm10"]:
        stats[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median()),
            "p10": float(df[col].quantile(0.10)),
            "p90": float(df[col].quantile(0.90)),
        }
    ratio = (df["pm10"] / df["pm2_5"].replace(0, np.nan)).dropna()
    stats["pm_ratio"] = {
        "mean": float(ratio.mean()),
        "std": float(ratio.std()),
        "min": float(ratio.min()),
        "max": float(ratio.max()),
        "p10": float(ratio.quantile(0.10)),
        "p90": float(ratio.quantile(0.90)),
    }
    stats["spike_fraction"] = float((df["pm2_5"] > 35).mean())
    return stats


def generate_temperature(n: int, records_per_day: int, stats: dict, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t_mean = stats["temperature_c"]["mean"]
    t_amp = (stats["temperature_c"]["max"] - stats["temperature_c"]["min"]) / 2.0 * 0.7

    # Time-of-day: warmest ~14:00, coolest ~05:00
    hour_of_day = np.arange(n) % records_per_day
    hour_frac = hour_of_day / records_per_day
    diurnal = t_amp * np.sin(2 * np.pi * (hour_frac - 0.35))

    # Low-frequency drift over 5 months (cooler start, warmer end)
    t = np.linspace(0, 1, n)
    drift = 1.5 * np.sin(2 * np.pi * t) + 0.8 * t

    # Bounded random walk for weather fronts
    noise = np.zeros(n)
    noise[0] = rng.normal(0, 0.3)
    for i in range(1, n):
        noise[i] = 0.98 * noise[i - 1] + rng.normal(0, 0.25)

    temp = t_mean + diurnal + drift + noise
    # Soft clip to observed range ±2
    t_min = stats["temperature_c"]["min"] - 2
    t_max = stats["temperature_c"]["max"] + 2
    temp = np.clip(temp, t_min, t_max)
    return temp


def generate_humidity(n: int, temperature: np.ndarray, stats: dict, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 1)
    h_mean = stats["humidity_percent"]["mean"]

    # Anti-correlated with temperature
    temp_norm = (temperature - temperature.mean()) / (temperature.std() + 1e-6)
    humidity = h_mean - 3.0 * temp_norm + rng.normal(0, 2.5, n)

    # Smooth humidity with AR
    h_smooth = np.zeros(n)
    h_smooth[0] = humidity[0]
    for i in range(1, n):
        h_smooth[i] = 0.95 * h_smooth[i - 1] + 0.05 * humidity[i]

    h_smooth = np.clip(h_smooth, stats["humidity_percent"]["min"], stats["humidity_percent"]["max"])
    return h_smooth


def generate_pm25(
    n: int,
    temperature: np.ndarray,
    humidity: np.ndarray,
    records_per_day: int,
    stats: dict,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed + 2)
    pm_mean = stats["pm2_5"]["mean"]
    pm_std = stats["pm2_5"]["std"]

    # Base level with slow drift
    base = pm_mean + 0.5 * pm_std * np.sin(2 * np.pi * np.arange(n) / (records_per_day * 14))

    # Weather effects: higher PM when hot/dry, lower when cool/humid
    temp_norm = (temperature - temperature.mean()) / (temperature.std() + 1e-6)
    hum_norm = (humidity - humidity.mean()) / (humidity.std() + 1e-6)
    weather_effect = 0.4 * pm_std * temp_norm - 0.3 * pm_std * hum_norm

    # Diurnal pattern: morning/evening peaks
    hour_of_day = np.arange(n) % records_per_day
    hour_frac = hour_of_day / records_per_day
    diurnal = 0.25 * pm_std * (
        np.sin(2 * np.pi * (hour_frac - 0.25)) + 0.5 * np.sin(2 * np.pi * (hour_frac - 0.70))
    )

    # AR(1) smooth evolution
    ar_noise = np.zeros(n)
    ar_noise[0] = rng.normal(0, 0.5)
    for i in range(1, n):
        ar_noise[i] = 0.85 * ar_noise[i - 1] + rng.normal(0, 0.6)

    pm = base + weather_effect + diurnal + ar_noise

    # Add pollution episodes
    n_episodes = int(n / records_per_day / 10)  # ~1 episode every 10 days
    for _ in range(n_episodes):
        start = rng.integers(int(0.05 * n), int(0.95 * n))
        duration = rng.integers(records_per_day * 0.75, records_per_day * 4)  # 18-96 h
        peak = rng.uniform(2.0, 4.0) * pm_std
        shape = np.linspace(0, 1, duration)
        shape = np.sin(np.pi * shape)  # rise then fall
        end = min(start + duration, n)
        pm[start:end] += peak * shape[: end - start]

    # Small random sensor noise
    pm += rng.normal(0, 0.15, n)

    # Keep within realistic bounds but allow a slightly wider max for 5-month variety
    pm = np.clip(pm, stats["pm2_5"]["min"], stats["pm2_5"]["max"] + 15)
    return pm


def generate_pm10(pm25: np.ndarray, stats: dict, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 3)
    ratio_mean = stats["pm_ratio"]["mean"]
    ratio_std = stats["pm_ratio"]["std"]

    # Ratio depends weakly on PM level: lower ratio at very high PM (more fine-dominated)
    pm_norm = (pm25 - pm25.mean()) / (pm25.std() + 1e-6)
    ratio = ratio_mean - 0.05 * pm_norm + rng.normal(0, ratio_std * 0.6, len(pm25))
    ratio = np.clip(ratio, stats["pm_ratio"]["min"], stats["pm_ratio"]["max"])

    pm10 = pm25 * ratio
    # Ensure PM10 >= PM2.5 (minimum ratio 1.05)
    pm10 = np.maximum(pm10, pm25 * 1.05)
    return pm10


def main():
    root = Path(__file__).resolve().parents[1]
    src_path = root / "data" / "raw" / "air_quality_data.csv"
    out_path = root / "data" / "raw" / "air_quality_data_5m.csv"

    stats = load_original_stats(src_path)
    (root / "data" / "raw" / "synthetic_5m_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    # 5 months at 5-minute resolution
    records_per_day = 288
    n_days = 150
    n = n_days * records_per_day

    # 5-month window entirely in the past relative to the system clock
    start_time = pd.Timestamp("2025-11-01 00:00:00")
    timestamps = pd.date_range(start=start_time, periods=n, freq="5min")

    temperature = generate_temperature(n, records_per_day, stats, seed=42)
    humidity = generate_humidity(n, temperature, stats, seed=42)
    pm25 = generate_pm25(n, temperature, humidity, records_per_day, stats, seed=42)
    pm10 = generate_pm10(pm25, stats, seed=42)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "temperature_c": np.round(temperature, 2),
        "humidity_percent": np.round(humidity, 2),
        "pm2_5": np.round(pm25, 2),
        "pm10": np.round(pm10, 2),
    })

    df.to_csv(out_path, index=False)
    print(f"Generated 5-month dataset: {out_path}")
    print(f"Rows: {len(df)}, Days: {len(df)/records_per_day:.1f}")
    print(f"PM2.5 > 35 fraction: {(df['pm2_5'] > 35).mean():.4f}")
    print(f"PM10 >= PM2.5: {(df['pm10'] >= df['pm2_5']).all()}")
    print(df[["temperature_c", "humidity_percent", "pm2_5", "pm10"]].describe())


if __name__ == "__main__":
    main()
