from core.pipeline.analysis_pipeline import run_full_analysis
import json

def find_non_serializable(obj, path="root"):
    import inspect

    if inspect.iscoroutine(obj):
        print(f"❌ Coroutine found at: {path}")
        return

    if isinstance(obj, dict):
                for k, v in obj.items():
                    find_non_serializable(v, f"{path}.{k}")

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            find_non_serializable(v, f"{path}[{i}]")

def to_serializable(obj):
    import numpy as np

    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]

    elif isinstance(obj, (np.integer,)):
        return int(obj)

    elif isinstance(obj, (np.floating,)):
        return float(obj)

    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()

    return obj

result = run_full_analysis("dunno3.wav")

find_non_serializable(result)

with open("output.json", "w") as f:
    json.dump(to_serializable(result), f, indent=2)
    