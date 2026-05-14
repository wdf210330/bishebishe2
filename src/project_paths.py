from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_LOG_DIR = PROJECT_ROOT / "debug_logs"


def ensure_debug_log_dir():
    DEBUG_LOG_DIR.mkdir(parents=True, exist_ok=True)
    return DEBUG_LOG_DIR
