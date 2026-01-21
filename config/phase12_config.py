"""Phase12 default configuration flags (disabled by default)."""
DEFAULTS = {
    "execution_realism": {"enabled": False, "seed": 42},
    "stress_engine": {"enabled": False, "seed": 42},
    "walk_forward": {"enabled": False, "seed": 42},
    "predictive_risk": {"enabled": False, "seed": 42},
    "performance_attribution": {"enabled": False, "seed": 42},
}

def get(module_name: str, key: str, default=None):
    return DEFAULTS.get(module_name, {}).get(key, default)
