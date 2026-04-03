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

result = run_full_analysis("dunno3.wav")

find_non_serializable(result)

with open("output.json", "w") as f:
    json.dump(result, f, indent=2)
    