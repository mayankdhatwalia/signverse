import argparse
import json

from .core import load_config


def main():
    parser = argparse.ArgumentParser(description="SignVerse AI collection, validation, training and inference")
    parser.add_argument("--config", default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--label", required=True)
    collect.add_argument("--participant", required=True)
    collect.add_argument("--session", required=True)
    collect.add_argument("--data", default="data")
    collect.add_argument("--camera", type=int, default=0)
    validate = commands.add_parser("validate")
    validate.add_argument("--data", default="data")
    training = commands.add_parser("train")
    training.add_argument("--data", default="data")
    training.add_argument("--output", default="artifacts")
    training.add_argument("--epochs", type=int, default=80)
    training.add_argument("--batch-size", type=int, default=16)
    training.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    training.add_argument("--split", choices=["session", "participant"], default="session")
    predict = commands.add_parser("predict")
    predict.add_argument("--model", required=True)
    predict.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.command == "collect":
            from .capture import webcam
            from .data import safe_name
            safe_name(args.participant)
            safe_name(args.session)
            if args.label not in config["labels"]:
                raise ValueError(f"Choose one of {config['labels']}")
            webcam(config, args.camera, args.data, args.label, args.participant, args.session)
        elif args.command == "predict":
            from .capture import webcam
            webcam(config, args.camera, model=args.model)
        elif args.command == "validate":
            from .data import load_dataset, split_dataset
            _, rows = load_dataset(args.data, config)
            print(rows.groupby("label").size().reindex(config["labels"], fill_value=0).to_string())
            split_dataset(rows, config["labels"])
            print("Validation passed, including session-separated split feasibility.")
            counts = rows.groupby("label").size().reindex(config["labels"], fill_value=0)
            if (counts < 40).any():
                print("Collection target pending: record 40–50 independent samples per class.")
        else:
            from .train import train
            run, metrics = train(args.data, args.output, config, args.epochs, args.batch_size, args.device, args.split)
            print(json.dumps({"run": str(run), "test_accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}, indent=2))
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"SignVerse: {error}\n")


if __name__ == "__main__":
    main()
