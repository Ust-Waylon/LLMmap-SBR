"""Run with python -m llmmapsbr from the release directory."""
import argparse
import json
import logging
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="LLMmap: frozen language representations for session recommendation")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Prepare first-day Amazon sessions and item cards")
    prepare.add_argument("--interactions", required=True, type=Path)
    prepare.add_argument("--metadata", required=True, type=Path)
    prepare.add_argument("--category", required=True, choices=["Cell_Phones_and_Accessories", "Grocery_and_Gourmet_Food", "Movies_and_TV"])
    prepare.add_argument("--output", required=True, type=Path)
    encode = commands.add_parser("encode", help="Cache frozen item-card language representations")
    encode.add_argument("--data-dir", required=True, type=Path)
    encode.add_argument("--output", required=True, type=Path)
    encode.add_argument("--device", default="cuda")
    encode.add_argument("--batch-size", type=int, default=16)
    encode.add_argument("--model-name", default="Qwen/Qwen3-Embedding-4B", help="Hub model ID or a local copy of that model")
    train = commands.add_parser("train", help="Train using prefixes and select by last-click validation HR@20")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--data-dir", required=True, type=Path)
    train.add_argument("--embeddings-dir", required=True, type=Path)
    train.add_argument("--output", required=True, type=Path)
    train.add_argument("--device", default="cuda")
    evaluate = commands.add_parser("evaluate", help="Evaluate a selected checkpoint on last-click sessions")
    evaluate.add_argument("--checkpoint", required=True, type=Path)
    evaluate.add_argument("--data-dir", required=True, type=Path)
    evaluate.add_argument("--embeddings-dir", required=True, type=Path)
    evaluate.add_argument("--split", choices=["valid", "test"], default="test")
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--output", type=Path, help="Optional new JSON metrics file")
    args = vars(parser.parse_args())
    command = args.pop("command")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if command == "prepare":
        from .data import prepare
        prepare(args.pop("interactions"), args.pop("metadata"), **args)
    elif command == "encode":
        from .encode import encode
        encode(**args)
    elif command == "train":
        from .train import train
        args["config_path"] = args.pop("config")
        train(**args)
    else:
        from .train import evaluate_checkpoint
        from .io import write_json
        output = args.pop("output")
        if output and output.exists():
            raise FileExistsError(f"Output already exists: {output}")
        result = evaluate_checkpoint(**args)
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            write_json(output, result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
