
import json
import sys
from pathlib import Path

def sum_values(data):
    """Recursively sums all numeric values in a dictionary."""
    total = 0
    if isinstance(data, dict):
        for value in data.values():
            total += sum_values(value)
    elif isinstance(data, (int, float)):
        return data
    return total

def split_json(file_path: Path):
    """
    Parses a JSON file, splits its top-level keys into separate files,
    and prints the sum of values for each new file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    output_dir = file_path.parent
    key_map = {
        "characters": "monograms.json",
        "bigrams": "bigrams.json",
        "trigrams": "trigrams.json",
        "skipgrams": "skipgrams.json"
    }
    other_data = {}
    created_files = []

    for key, value in data.items():
        if key in key_map:
            output_filename = key_map[key]
            output_path = output_dir / output_filename
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(value, f, indent=2)
            created_files.append(output_path)
        else:
            other_data[key] = value

    if other_data:
        output_path = output_dir / "others.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(other_data, f, indent=2)
        created_files.append(output_path)

    print("Sum of values for each created file:")
    print("-" * 40)
    for path in sorted(created_files):
        with open(path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        total_sum = sum_values(content)
        print(f"{path.name:<20} | Sum: {total_sum:,.2f}")
    print("-" * 40)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <input_json_file>", file=sys.stderr)
        sys.exit(1)

    input_file = Path(sys.argv[1])
    split_json(input_file)
