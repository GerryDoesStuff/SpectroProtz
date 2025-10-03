from datetime import datetime
import platform

def start_audit() -> list[str]:
    return [f"Session start: {datetime.now().isoformat()}",
            f"Platform: {platform.platform()}" ]

def log_step(audit: list[str], msg: str):
    audit.append(msg)
