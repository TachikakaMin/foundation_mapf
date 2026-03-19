import argparse
import os
import sys

import yaml


def _build_parser():
    parser = argparse.ArgumentParser(description="UNet training")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="path to YAML config file; CLI args override config values",
    )

    return parser


def _config_cli_has_extra_args(argv):
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token == "--config":
            skip_next = True
            continue
        if token.startswith("--config="):
            continue
        return True
    return False


def _normalize_config_values(config_data):
    normalized = dict(config_data)
    if "learning_rate" in normalized:
        if "lr" in normalized:
            raise ValueError("config cannot contain both 'learning_rate' and 'lr'")
        normalized["lr"] = normalized.pop("learning_rate")
    sample_paths = normalized.get("sample_data_path")
    if isinstance(sample_paths, str):
        normalized["sample_data_path"] = [sample_paths]
    return normalized


def _add_bool_argument(parser, name, default, help_text):
    dest = name.lstrip("-").replace("-", "_")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(name, dest=dest, action="store_true", help=help_text)
    group.add_argument(
        f"--no-{dest.replace('_', '-')}",
        dest=dest,
        action="store_false",
        help=f"Disable {help_text.lower()}",
    )
    parser.set_defaults(**{dest: default})


def _load_config_defaults(parser):
    partial_args, _ = parser.parse_known_args()
    if not partial_args.config:
        return {}

    config_path = os.path.abspath(partial_args.config)
    if not os.path.isfile(config_path):
        parser.error(f"config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file) or {}

    if not isinstance(config_data, dict):
        parser.error(f"config file must contain a top-level mapping: {config_path}")

    try:
        config_data = _normalize_config_values(config_data)
    except ValueError as error:
        parser.error(str(error))
    valid_keys = {action.dest for action in parser._actions}
    unknown_keys = sorted(set(config_data) - valid_keys)
    if unknown_keys:
        parser.error(
            f"unknown config keys in {config_path}: {', '.join(unknown_keys)}"
        )

    return config_data


def get_args():
    """用于解析命令行参数"""
    parser = _build_parser()

    parser.add_argument("--seed", "-sd", type=int, default=1919180, help="seed")
    parser.add_argument("--log_dir", "-ld", type=str, default="runs", help="plot log")

    # dataset
    parser.add_argument(
        "--dataset_path",
        "-dp",
        type=str,
        default="data/input_data",
        help="offline dataset path (.mbin files)",
    )
    parser.add_argument(
        "--val_dataset_path",
        type=str,
        default=None,
        help="offline validation dataset path (.mbin files), defaults to --dataset_path",
    )
    parser.add_argument(
        "--dataset_mode",
        type=str,
        choices=["offline", "online"],
        default="offline",
        help="training dataset mode: offline=.mbin, online=on-the-fly generation",
    )
    parser.add_argument(
        "--train_map_path",
        type=str,
        default="data/map_files",
        help="map directory for online training mode",
    )
    parser.add_argument(
        "--online_total_steps",
        type=int,
        default=200000,
        help="total optimizer steps in online mode",
    )
    parser.add_argument(
        "--online_eval_interval_steps",
        type=int,
        default=4000,
        help="run validation every N optimizer steps in online mode",
    )
    parser.add_argument(
        "--online_save_interval_steps",
        type=int,
        default=4000,
        help="save checkpoint every N optimizer steps in online mode",
    )
    parser.add_argument(
        "--online_inference_test_interval_steps",
        type=int,
        default=4000,
        help="run inference test every N optimizer steps in online mode",
    )
    parser.add_argument(
        "--online_time_limit_sec",
        type=int,
        default=2,
        help="time limit for each online scenario generation",
    )
    parser.add_argument(
        "--online_retry_limit",
        type=int,
        default=20,
        help="max retries for online scenario generation",
    )
    parser.add_argument(
        "--online_buffer_size",
        type=int,
        default=512,
        help="global online step-buffer size per map-size group",
    )
    parser.add_argument(
        "--online_buffer_timeout_sec",
        type=float,
        default=1.0,
        help="queue wait timeout (seconds) for the online step buffer consumer",
    )
    parser.add_argument(
        "--sample_data_path",
        "-sp",
        type=str,
        nargs="+",
        default=["data/input_data/maze-32-32-10-1-75/maze-32-32-10-1-75-0-16.mbin"
                 ],
        help="fixed offline sample .mbin files used by inference test during training",
    )
    parser.add_argument(
        "--inference_num_cases",
        type=int,
        default=1,
        help="number of fixed sample cases to run for inference test",
    )
    parser.add_argument(
        "--inference_test_interval",
        type=int,
        default=0,
        help="offline mode only: run inference test every N epochs; 0 means reuse --eval_interval",
    )
    parser.add_argument(
        "--inference_action_choice",
        type=str,
        choices=["sample", "max"],
        default="max",
        help="action selection used during inference test rollout",
    )
    parser.add_argument(
        "--num_workers",
        "-nw",
        type=int,
        default=20,
        help="offline: DataLoader workers; online: step-buffer producer threads and eval/sample DataLoader workers",
    )
    # training
    parser.add_argument(
        "--epochs", "-ep", type=int, default=100, help="offline mode only: number of epochs"
    )
    parser.add_argument("--batch_size", "-bs", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=1e-5,
        help="Learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--weight_decay", "-wd", type=float, default=1e-3, help="weight decay"
    )
    parser.add_argument("--eval_interval", type=int, default=2, help="offline mode only: eval interval in epochs")
    parser.add_argument("--save_interval", type=int, default=2, help="offline mode only: save interval in epochs")
    # model
    parser.add_argument(
        "--feature_dim", "-fd", type=int, default=6, help="feature dimension"
    )
    parser.add_argument("--feature_type", "-ft", type=str, default="gradient", help="feature type")
    parser.add_argument(
        "--steps",
        "-st",
        type=int,
        default=300,
        help="maximum rollout steps for inference test / path_formation",
    )
    parser.add_argument("--action_dim", "-ad", type=int, default=5, help="Action types")
    parser.add_argument("--model", "-m", type=str, default="unet", help="Model type")
    _add_bool_argument(parser, "--bilinear", False, "Use bilinear upsampling")
    parser.add_argument("--first_layer_channels", "-flc", type=int, default=64, help="First layer channels")
    parser.add_argument("--blocks_per_stage", type=int, default=1, help="ResBlocks per UNet stage (0=legacy DoubleConv)")
    # 添加分布式训练参数
    _add_bool_argument(parser, "--distributed", False, "Enable distributed training")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model file")

    config_defaults = _load_config_defaults(parser)
    if config_defaults:
        parser.set_defaults(**config_defaults)

    args = parser.parse_args()

    if args.config:
        config_path = os.path.abspath(args.config)
        has_cli_overrides = _config_cli_has_extra_args(sys.argv[1:])
        args.config_source = (
            f"{config_path} + CLI overrides" if has_cli_overrides else config_path
        )
    else:
        args.config_source = "CLI only"

    return args


if __name__ == "__main__":
    args = get_args()
