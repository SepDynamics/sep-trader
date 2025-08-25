import json
import numpy as np

def check_json_serializable(obj, path=""):
    """Recursively check if an object is JSON serializable."""
    try:
        json.dumps(obj)
        return True, None
    except Exception as e:
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_path = f"{path}.{k}" if path else k
                serializable, error = check_json_serializable(v, new_path)
                if not serializable:
                    return False, error
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                new_path = f"{path}[{i}]" if path else f"[{i}]"
                serializable, error = check_json_serializable(v, new_path)
                if not serializable:
                    return False, error
        else:
            return False, f"Path: {path}, Type: {type(obj)}, Value: {obj}"
        return True, None

# Test with a sample dict that might have the issue
test_dict = {
    'bool_val': np.bool_(True),
    'float_val': np.float64(1.0),
    'int_val': np.int64(1),
    'nested': {
        'bool_val': np.bool_(False),
        'list_val': [np.bool_(True), np.float64(2.0)]
    }
}

serializable, error = check_json_serializable(test_dict)
print(f"Serializable: {serializable}")
if error:
    print(f"Error: {error}")

# Test individual values
print("\nIndividual type tests:")
values = [
    np.bool_(True),
    np.float64(1.0),
    np.int64(1),
    True,
    1.0,
    1
]

for val in values:
    try:
        json.dumps(val)
        print(f"{type(val)}: OK")
    except Exception as e:
        print(f"{type(val)}: ERROR - {e}")