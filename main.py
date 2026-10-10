from neural_network import NeuralNetwork
from dataset import load_dataset, normalize_dataset
import argparse
from pathlib import Path
from math_helper import sigmoid, mse

def execute_inference(neural_network: NeuralNetwork):
    playing_speed = float(input("playing_speed: "))
    pitch_entropy = float(input("pitch_entropy: "))
    average_pitch = float(input("average_pitch: "))
    pitch_range = float(input("pitch_range: "))
    bpm = float(input("bpm: "))
    average_note_duration = float(input("average_note_duration: "))
    shortest_rhythm_value = float(input("shortest_rhythm_value: "))
    duration_deviation = float(input("duration_deviation: "))
    pitch_value_deviation = float(input("pitch_value_deviation: "))
    pitch_class_deviation = float(input("pitch_class_deviation: "))
    key_signature = float(input("key_signature: "))

    inputs = [
        playing_speed,
        pitch_entropy,
        average_pitch,
        pitch_range,
        bpm,
        average_note_duration,
        shortest_rhythm_value,
        duration_deviation,
        pitch_value_deviation,
        pitch_class_deviation,
        key_signature
    ]
    
    outputs = neural_network.predict(inputs)
    print(f"Estimated difficulty is {outputs[0] * 10}")

def load_neural_network(model_path: Path) -> NeuralNetwork:
    if model_path.exists():
        neural_network = NeuralNetwork.load(model_path)
    else:
        LAYERS = [11, 4, 4, 1]
        neural_network = NeuralNetwork(LAYERS, sigmoid, sigmoid, mse, 0.0001)

    return neural_network

def train(neural_network: NeuralNetwork) -> None:
    x_train, y_train = load_dataset("dataset/normalized.csv")
    x_train = [[float(sample) for sample in x] for x in x_train]
    y_train = [[float(y) / 10] for y in y_train]
    
    print(f"Average Error: {neural_network.evaluate(x_train, y_train)}")
    neural_network.stochastic_train(x_train, y_train, 100)
    print(f"Average Error: {neural_network.evaluate(x_train, y_train)}")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("command", choices=["normalize", "inference", "train"])
    parser.add_argument("--model")

    args = parser.parse_args()

    if args.command in ["inference", "train"] and not args.model:
        parser.error(f"--model is required for '{args.command}'")

    if args.command == "normalize":
        normalize_dataset("dataset/features.csv", "dataset/normalized.csv")
        return

    model_path = Path(f"models/{args.model}.pkl")
    neural_network = load_neural_network(model_path)

    if args.command == "inference":
        execute_inference(neural_network)
    elif args.command == "train":
        train(neural_network)
        neural_network.save(str(model_path))

if __name__ == "__main__":
    main()