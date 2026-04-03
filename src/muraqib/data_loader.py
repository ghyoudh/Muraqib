import os
import numpy as np
import pandas as pd

_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "saudi_construction_activities.csv"
)

_COMPLEXITY_MAP = {"Low": 0, "Medium": 1, "High": 2}
_WEATHER_MAP = {"Low": 0, "Medium": 1, "High": 2}

# Seed for reproducibility — same "synthetic" data every run
_RNG_SEED = 42


def _derive_delay(row) -> int:
    """Business-rule delay label derived from enriched features."""
    delayed = False
    if row["complexity_enc"] == 2 and row["supply_delay_days"] > 10:
        delayed = True
    if row["subcontractor_performance"] < 5:
        delayed = True
    if row["weather_enc"] == 2 and row["labor_availability"] < 70:
        delayed = True
    return int(delayed)


def load_data() -> pd.DataFrame:
    """Load CSV, synthetically enrich it, and return the full DataFrame."""
    df = pd.read_csv(_DATA_PATH)

    # Drop any trailing empty rows
    df.dropna(how="all", inplace=True)
    df.reset_index(drop=True, inplace=True)

    rng = np.random.default_rng(_RNG_SEED)
    n = len(df)

    # Synthetic realistic features
    df["supply_delay_days"] = rng.integers(0, 31, size=n)
    df["subcontractor_performance"] = np.round(rng.uniform(2.0, 10.0, size=n), 1)
    df["weather_risk"] = rng.choice(["Low", "Medium", "High"], size=n, p=[0.4, 0.4, 0.2])
    df["labor_availability"] = np.round(rng.uniform(50.0, 100.0, size=n), 1)

    # Encoded versions for the model
    df["complexity_enc"] = df["Complexity Level"].map(_COMPLEXITY_MAP).fillna(0).astype(int)
    df["weather_enc"] = df["weather_risk"].map(_WEATHER_MAP).fillna(0).astype(int)

    # Target label
    df["is_delayed"] = df.apply(_derive_delay, axis=1)

    # Parse start date
    df["Expected Start Date"] = pd.to_datetime(df["Expected Start Date"])
    df["start_month"] = df["Expected Start Date"].dt.month

    return df


def get_feature_columns() -> list:
    return [
        "complexity_enc",
        "supply_delay_days",
        "subcontractor_performance",
        "weather_enc",
        "labor_availability",
    ]


def get_feature_display_names(lang: str = "en") -> dict:
    if lang == "ar":
        return {
            "complexity_enc": "مستوى التعقيد",
            "supply_delay_days": "تأخير التوريد (أيام)",
            "subcontractor_performance": "أداء المقاول الباطن",
            "weather_enc": "مخاطر الطقس",
            "labor_availability": "توافر العمالة (%)",
        }
    return {
        "complexity_enc": "Complexity Level",
        "supply_delay_days": "Supply Delay (days)",
        "subcontractor_performance": "Subcontractor Performance",
        "weather_enc": "Weather Risk",
        "labor_availability": "Labor Availability (%)",
    }
