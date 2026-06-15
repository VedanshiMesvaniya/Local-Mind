import psutil
import platform

def get_current_performance_metrics():
    metrics_snapshot = {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
    }
    return metrics_snapshot