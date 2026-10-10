import csv
import math

def load_dataset(path: str):
    extracted_x = []
    extracted_y = []

    with open(path, "r", encoding="utf-8") as dataset_stream:
        reader = csv.DictReader(dataset_stream)

        for row in reader:
            sample_y = row["ps_grade"]
            sample_x = [
                row["playing_speed"],
                row["pitch_entropy"],
                row["average_pitch"],
                row["pitch_range"],
                row["bpm"],
                row["average_note_duration"],
                row["shortest_rhythm_value"],
                row["duration_deviation"],
                row["pitch_value_deviation"],
                row["pitch_class_deviation"],
                row["key_signature"]
            ]

            extracted_x.append(sample_x)
            extracted_y.append(sample_y)

    return (extracted_x, extracted_y)

def normalize_dataset(input: str, output: str) -> None:
    extracted_x, extracted_y = load_dataset(input)
    feature_mins = [math.inf] * 11
    feature_maxs = [-math.inf] * 11
    for sample in extracted_x:
        for i, feature in enumerate(sample):
            feature_mins[i] = min(feature_mins[i], float(feature))
            feature_maxs[i] = max(feature_maxs[i], float(feature))

    normalized_x = []
    for sample in extracted_x:
        normalized_features = []

        for i, feature in enumerate(sample):
            normalized_feature = minmax_normalization(float(feature), feature_mins[i], feature_maxs[i])
            normalized_features.append(normalized_feature)

        normalized_x.append(normalized_features)

    HEADERS = [
        "playing_speed",
        "pitch_entropy",
        "average_pitch",
        "pitch_range",
        "bpm",
        "average_note_duration",
        "shortest_rhythm_value",
        "duration_deviation",
        "pitch_value_deviation",
        "pitch_class_deviation",
        "key_signature",
        "ps_grade"
    ]

    with open(output, "w", newline="", encoding="utf-8") as output_stream:
        writer = csv.writer(output_stream)
        writer.writerow(HEADERS)

        for x, y in zip(normalized_x, extracted_y):
            writer.writerow(x + [y])

def minmax_normalization(x, x_min, x_max):
    return float((x - x_min) / (x_max - x_min))
