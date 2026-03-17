import glob
import inspect
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from MAPF_dataset_mbin import MAPFDataset
from models.CNN import CNN
from models.unet import UNet
from tools.path_formation import path_formation
from train_args import get_args


def get_map_dims(file_path):
    parts = os.path.basename(file_path).split("-")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse map dims from file name: {file_path}")
    return int(parts[1]), int(parts[2])


def group_files_by_dims(file_paths, min_map_size=32):
    dimension_groups = {}
    for file_path in sorted(file_paths):
        try:
            dims = get_map_dims(file_path)
        except Exception:
            continue
        if dims[0] < min_map_size or dims[1] < min_map_size:
            continue
        dimension_groups.setdefault(dims, []).append(file_path)
    return dimension_groups


def create_offline_validation_loaders(args):
    val_root = args.val_dataset_path if args.val_dataset_path else args.dataset_path
    val_mbin_files = sorted(glob.glob(os.path.join(val_root, "**/*.mbin"), recursive=True))
    if not val_mbin_files:
        raise RuntimeError(f"❌ 未找到验证集.mbin文件: {val_root}")

    dimension_groups = group_files_by_dims(val_mbin_files, min_map_size=32)
    if not dimension_groups:
        raise RuntimeError("❌ 验证集中没有找到合适大小(>=32x32)的地图数据")

    val_loaders = {}
    selected_val_files = []
    use_all_validation_files = args.val_dataset_path is not None

    for dims, files in dimension_groups.items():
        if use_all_validation_files:
            val_list = files
        else:
            n_val = int(0.1 * len(files))
            if n_val <= 0:
                if args.local_rank == 0:
                    print(f"跳过验证分组 {dims}, 文件太少: {len(files)}")
                continue
            val_list = files[:n_val]
        selected_val_files.extend(val_list)

        val_data = MAPFDataset(val_list, args.feature_dim, args.feature_type)
        val_sampler = (
            torch.utils.data.distributed.DistributedSampler(val_data, shuffle=False)
            if args.distributed
            else None
        )
        val_loaders[dims] = DataLoader(
            val_data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sampler=val_sampler,
        )

    if not val_loaders:
        raise RuntimeError("❌ 未能创建任何有效验证DataLoader")

    return val_loaders, selected_val_files


def create_offline_train_loaders(args):
    mbin_files = sorted(glob.glob(os.path.join(args.dataset_path, "**/*.mbin"), recursive=True))
    if not mbin_files:
        raise RuntimeError("❌ 未找到.mbin文件, 请先运行数据转换")

    dimension_groups = group_files_by_dims(mbin_files, min_map_size=32)
    if not dimension_groups:
        raise RuntimeError("❌ 没有找到合适大小(>=32x32)的训练数据")

    train_loaders = {}
    train_loader_weights = {}

    for dims, files in dimension_groups.items():
        n_test = int(0.1 * len(files))
        train_list = files[n_test:]
        if not train_list:
            if args.local_rank == 0:
                print(f"跳过训练分组 {dims}, 可用训练文件为0")
            continue

        train_data = MAPFDataset(train_list, args.feature_dim, args.feature_type)
        train_sampler = (
            torch.utils.data.distributed.DistributedSampler(train_data)
            if args.distributed
            else None
        )
        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sampler=train_sampler,
        )
        train_loaders[dims] = train_loader
        train_loader_weights[dims] = len(train_list)

        if args.local_rank == 0:
            print(f"Offline train group {dims}: {len(train_list)} files")

    if not train_loaders:
        raise RuntimeError("❌ 未能创建任何有效训练DataLoader")

    return train_loaders, train_loader_weights


def _filter_constructor_kwargs(constructor, kwargs):
    target = constructor
    if inspect.isclass(constructor):
        target = constructor.__init__

    signature = inspect.signature(target)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def create_online_train_loaders(args):
    from MAPF_online_dataset import MAPFOnlineDataset

    map_files = sorted(glob.glob(os.path.join(args.train_map_path, "**/*.map"), recursive=True))
    if not map_files:
        raise RuntimeError(f"❌ 未找到在线训练地图文件: {args.train_map_path}")

    dimension_groups = group_files_by_dims(map_files, min_map_size=32)
    if not dimension_groups:
        raise RuntimeError("❌ 在线训练地图中没有合适大小(>=32x32)的地图")

    train_loaders = {}
    train_loader_weights = {}
    world_size = dist.get_world_size() if args.distributed else 1

    for dims, files in dimension_groups.items():
        dataset_kwargs = {
            "time_limit_sec": args.online_time_limit_sec,
            "retry_limit": args.online_retry_limit,
            "seed": args.seed + args.local_rank * 1000003,
            "first_step": False,
            # integration hook: allow dataset implementations to shard by rank
            "rank": args.local_rank,
            "world_size": world_size,
        }
        dataset_kwargs = _filter_constructor_kwargs(MAPFOnlineDataset, dataset_kwargs)
        train_data = MAPFOnlineDataset(
            files,
            args.feature_dim,
            args.feature_type,
            **dataset_kwargs,
        )
        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        train_loaders[dims] = train_loader
        pair_weight = sum(getattr(pair, "weight", 0) for pair in getattr(train_data, "_pairs", []))
        train_loader_weights[dims] = pair_weight if pair_weight > 0 else len(files)

        if args.local_rank == 0:
            print(
                f"Online train group {dims}: {len(files)} maps, "
                f"sampling_weight={train_loader_weights[dims]}"
            )

    if not train_loaders:
        raise RuntimeError("❌ 未能创建任何有效在线训练DataLoader")

    return train_loaders, train_loader_weights


def get_online_schedule(args):
    schedule = {
        "total_steps": int(args.online_total_steps),
        "eval_interval_steps": int(args.online_eval_interval_steps),
        "save_interval_steps": int(args.online_save_interval_steps),
        "inference_interval_steps": int(args.online_inference_test_interval_steps),
    }

    if schedule["total_steps"] <= 0:
        raise ValueError("--online_total_steps 必须大于 0")
    if schedule["eval_interval_steps"] <= 0:
        raise ValueError("--online_eval_interval_steps 必须大于 0")
    if schedule["save_interval_steps"] <= 0:
        raise ValueError("--online_save_interval_steps 必须大于 0")
    if schedule["inference_interval_steps"] <= 0:
        raise ValueError("--online_inference_test_interval_steps 必须大于 0")

    return schedule

def get_model_stats(model):
    module = model.module if isinstance(model, DDP) else model
    total_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    model_memory_mb = total_params * 4 / (1024**2)
    return total_params, model_memory_mb


def format_kv_table(rows):
    if not rows:
        return ""
    key_width = max(len(str(key)) for key, _ in rows)
    return "\n".join(f"{str(key):<{key_width}} : {value}" for key, value in rows)


def summarize_loader_groups(loaders, weights=None):
    if not loaders:
        return "none"
    parts = []
    for dims in sorted(loaders):
        text = f"{dims[0]}x{dims[1]}"
        if weights is not None:
            text = f"{text} (weight={weights.get(dims, 0)})"
        parts.append(text)
    return ", ".join(parts)


def print_runtime_summary(
    args,
    model,
    device,
    train_loaders,
    train_loader_weights,
    val_loaders,
    sample_paths=None,
    online_schedule=None,
):
    if args.local_rank != 0:
        return

    total_params, model_memory_mb = get_model_stats(model)
    rows = [
        ("config_source", "CLI -> train_args.py -> train.py"),
        ("device", device),
        ("dataset_mode", args.dataset_mode),
        ("train_data", args.train_map_path if args.dataset_mode == "online" else args.dataset_path),
        ("val_data", args.val_dataset_path if args.val_dataset_path else args.dataset_path),
        ("train_groups", summarize_loader_groups(train_loaders, train_loader_weights)),
        ("val_groups", summarize_loader_groups(val_loaders)),
        ("model", args.model),
        ("feature", f"dim={args.feature_dim}, type={args.feature_type}, action_dim={args.action_dim}"),
        ("model_params", total_params),
        ("model_size_mb", f"{model_memory_mb:.2f}"),
        ("batch_size", args.batch_size),
        ("lr", args.lr),
        ("weight_decay", args.weight_decay),
        ("num_workers", args.num_workers),
        ("inference_cases", args.inference_num_cases),
        ("inference_action", args.inference_action_choice),
        ("inference_steps", args.steps),
    ]
    if sample_paths:
        rows.append(("inference_samples", ", ".join(os.path.basename(path) for path in sample_paths)))
    if args.dataset_mode == "online":
        rows.extend(
            [
                ("progress_unit", "optimizer_steps"),
                ("online_total_steps", online_schedule["total_steps"]),
                ("online_eval_interval_steps", online_schedule["eval_interval_steps"]),
                ("online_save_interval_steps", online_schedule["save_interval_steps"]),
                ("online_inference_interval_steps", online_schedule["inference_interval_steps"]),
                ("estimated_train_samples", online_schedule["total_steps"] * args.batch_size),
                ("online_time_limit_sec", args.online_time_limit_sec),
                ("online_retry_limit", args.online_retry_limit),
            ]
        )
    else:
        inference_interval = args.inference_test_interval if args.inference_test_interval > 0 else args.eval_interval
        rows.extend(
            [
                ("progress_unit", "epochs"),
                ("epochs", args.epochs),
                ("eval_interval", args.eval_interval),
                ("save_interval", args.save_interval),
                ("inference_interval", inference_interval),
            ]
        )

    summary = format_kv_table(rows)
    print("=== Runtime Config ===")
    print(summary)
    args.writer.add_text("RuntimeConfig", f"```text\n{summary}\n```", 0)


def evaluate_valid_loss(args, model, val_loader, loss_fn, device):
    model.eval()
    val_loss = 0.0
    total_agents = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", disable=args.local_rank != 0):
            feature = batch["feature"].to(device)
            action_y = batch["action"].to(device)
            mask = batch["mask"].to(device)

            logits, _ = model(feature)
            loss = loss_fn(logits, action_y)
            masked_loss = loss * mask.float()
            val_loss += masked_loss.detach().sum().item()
            total_agents += mask.detach().sum().item()

    if total_agents == 0:
        return float("inf")
    return val_loss / total_agents


def run_validation(args, model, val_loaders, loss_fn, device, log_step, progress_label):
    val_losses = []
    for dims, val_loader in val_loaders.items():
        val_loss = evaluate_valid_loss(args, model, val_loader, loss_fn, device)
        val_losses.append(val_loss)
        if args.local_rank == 0:
            args.writer.add_scalar(f"Loss/Val_{dims[0]}x{dims[1]}", val_loss, log_step)
            print(
                f"{progress_label}, Validation mean Loss "
                f"[{dims[0]}x{dims[1]}]: {val_loss}"
            )

    if args.local_rank == 0 and val_losses:
        aggregated_val_loss = float(sum(val_losses) / len(val_losses))
        args.writer.add_scalar("Loss/Val", aggregated_val_loss, log_step)
        print(f"{progress_label}, Validation mean Loss (aggregated): {aggregated_val_loss}")


def run_inference_test(args, model, sample_loader, device, log_step, progress_label):
    if args.local_rank != 0 or sample_loader is None:
        return

    metrics_list = []
    sample_count = len(sample_loader.dataset)
    for idx in range(sample_count):
        _, _, _, file_name, metrics = path_formation(
            model,
            sample_loader,
            idx,
            device,
            args.feature_type,
            action_choice=args.inference_action_choice,
            steps=args.steps,
            return_metrics=True,
        )
        metrics_list.append(metrics)

        for key, value in metrics.items():
            args.writer.add_scalar(f"Inference/{key}_{idx}", value, log_step)

        metric_str = ", ".join(f"{key}={value}" for key, value in metrics.items())
        print(f"Inference case {idx}: {os.path.basename(file_name)} | {metric_str}")

    if not metrics_list:
        return

    summary = {}
    for key in metrics_list[0]:
        summary[key] = float(sum(metrics[key] for metrics in metrics_list) / len(metrics_list))
        args.writer.add_scalar(f"InferenceSummary/{key}", summary[key], log_step)

    summary_str = ", ".join(f"{key}={value:.4f}" for key, value in summary.items())
    print(f"{progress_label}, Inference summary: {summary_str}")


def train_offline(args, model, train_loaders, val_loaders, sample_loader, optimizer, loss_fn, device):
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        total_agents = 0

        for train_loader in train_loaders.values():
            sampler = getattr(train_loader, "sampler", None)
            if args.distributed and sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            for _, batch in tqdm(
                enumerate(train_loader),
                desc=f"Epoch {epoch}/{args.epochs}",
                disable=args.local_rank != 0,
                total=len(train_loader),
            ):
                feature = batch["feature"].to(device)
                action_y = batch["action"].to(device)
                mask = batch["mask"].to(device)

                logit, _ = model(feature)
                loss = loss_fn(logit, action_y)
                masked_loss = loss * mask.float()
                averaged_loss = masked_loss.sum() / mask.sum()

                optimizer.zero_grad()
                averaged_loss.backward()
                optimizer.step()

                train_loss += masked_loss.detach().sum().item()
                total_agents += mask.detach().sum().item()

        train_loss = train_loss / total_agents if total_agents > 0 else float("inf")
        if args.local_rank == 0:
            args.writer.add_scalar("Loss/Train", train_loss, epoch)
            print(f"Epoch {epoch}/{args.epochs}, Training mean Loss: {train_loss}")

        if epoch % args.eval_interval == 0:
            run_validation(
                args,
                model,
                val_loaders,
                loss_fn,
                device,
                epoch,
                f"Epoch {epoch}/{args.epochs}",
            )

        inference_interval = args.inference_test_interval if args.inference_test_interval > 0 else args.eval_interval
        if inference_interval > 0 and epoch % inference_interval == 0:
            run_inference_test(
                args,
                model,
                sample_loader,
                device,
                epoch,
                f"Epoch {epoch}/{args.epochs}",
            )

        if epoch % args.save_interval == 0 and args.local_rank == 0:
            file_path = os.path.join(args.real_log_dir, f"model_checkpoint_epoch_{epoch}.pth")
            if args.distributed:
                torch.save(model.module.state_dict(), file_path)
            else:
                torch.save(model.state_dict(), file_path)


def train_online(args, model, train_loaders, train_loader_weights, val_loaders, sample_loader, optimizer, loss_fn, device):
    online_schedule = get_online_schedule(args)

    dims_list = sorted(train_loaders.keys())
    weights = np.array([max(train_loader_weights.get(dims, 0), 1) for dims in dims_list], dtype=np.float64)
    weights_sum = float(weights.sum())
    if weights_sum <= 0:
        weights = np.ones(len(dims_list), dtype=np.float64)
        weights_sum = float(weights.sum())
    probs = weights / weights_sum

    rng = np.random.default_rng(args.seed + args.local_rank * 1000003 + 17)
    train_iters = {dims: iter(loader) for dims, loader in train_loaders.items()}

    train_loss_window = 0.0
    total_agents_window = 0
    total_steps = online_schedule["total_steps"]

    for step in tqdm(
        range(1, total_steps + 1),
        desc="Online Train",
        disable=args.local_rank != 0,
        total=total_steps,
    ):
        dims = dims_list[int(rng.choice(len(dims_list), p=probs))]
        train_loader = train_loaders[dims]
        train_iter = train_iters[dims]
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            train_iters[dims] = train_iter
            batch = next(train_iter)

        feature = batch["feature"].to(device)
        action_y = batch["action"].to(device)
        mask = batch["mask"].to(device)

        logit, _ = model(feature)
        loss = loss_fn(logit, action_y)
        masked_loss = loss * mask.float()
        averaged_loss = masked_loss.sum() / mask.sum()

        optimizer.zero_grad()
        averaged_loss.backward()
        optimizer.step()

        train_loss_window += masked_loss.detach().sum().item()
        total_agents_window += mask.detach().sum().item()

        should_eval = (
            step % online_schedule["eval_interval_steps"] == 0 or step == total_steps
        )
        should_infer = (
            step % online_schedule["inference_interval_steps"] == 0 or step == total_steps
        )
        should_save = (
            step % online_schedule["save_interval_steps"] == 0 or step == total_steps
        )
        should_report = should_eval or should_infer or should_save or step == total_steps

        if should_report and args.local_rank == 0 and total_agents_window > 0:
            train_loss = train_loss_window / total_agents_window
            args.writer.add_scalar("Loss/Train", train_loss, step)
            print(f"Step {step}/{total_steps}, Training mean Loss: {train_loss}")
            train_loss_window = 0.0
            total_agents_window = 0

        progress_label = f"Step {step}/{total_steps}"
        if should_eval:
            run_validation(args, model, val_loaders, loss_fn, device, step, progress_label)

        if should_infer:
            run_inference_test(args, model, sample_loader, device, step, progress_label)

        if should_save and args.local_rank == 0:
            file_path = os.path.join(args.real_log_dir, f"model_checkpoint_step_{step}.pth")
            if args.distributed:
                torch.save(model.module.state_dict(), file_path)
            else:
                torch.save(model.state_dict(), file_path)


def train(args, model, train_loaders, train_loader_weights, val_loaders, sample_loader, optimizer, loss_fn, device):
    if args.dataset_mode == "online":
        train_online(
            args,
            model,
            train_loaders,
            train_loader_weights,
            val_loaders,
            sample_loader,
            optimizer,
            loss_fn,
            device,
        )
        return

    train_offline(
        args,
        model,
        train_loaders,
        val_loaders,
        sample_loader,
        optimizer,
        loss_fn,
        device,
    )

if __name__ == "__main__":

    # arguments
    args = get_args()

    # 设置分布式训练
    if args.distributed:
        dist.init_process_group(backend='nccl', init_method='env://')
        args.local_rank = dist.get_rank()
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda:{}".format(args.local_rank))
    else:
        args.local_rank = 0    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.real_log_dir = os.path.join(args.log_dir, f"{args.current_time}")
    
    # 只在主进程上创建tensorboard writer
    if args.local_rank == 0:
        args.writer = SummaryWriter(log_dir=args.real_log_dir)
        args_dict = vars(args)
        args_str = "\n".join([f"{key}: {value}" for key, value in args_dict.items()])
        args.writer.add_text("Args", args_str, 0)
        print(args_str)

    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # model
    if args.model == "unet":
        net = UNet(n_channels=args.feature_dim, n_classes=args.action_dim, first_layer_channels=args.first_layer_channels, bilinear=args.bilinear)
    elif args.model == "cnn":
        net = CNN(n_channels=args.feature_dim, n_classes=args.action_dim)
    if args.model_path:
        net.load_state_dict(torch.load(args.model_path, map_location=device))
    net.to(device)
    # 如果使用分布式训练, 将模型包装为DDP模型
    if args.distributed:
        net = DDP(net, device_ids=[args.local_rank])

    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),  # 默认值, 适合大多数情况
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    try:
        val_loaders, available_val_mbin = create_offline_validation_loaders(args)
        if args.dataset_mode == "online":
            train_loaders, train_loader_weights = create_online_train_loaders(args)
        else:
            train_loaders, train_loader_weights = create_offline_train_loaders(args)
    except RuntimeError as error:
        if args.local_rank == 0:
            print(str(error))
        raise SystemExit(1)

    sample_loader = None
    sample_candidates = []
    if args.inference_num_cases > 0:
        sample_candidates = [path for path in args.sample_data_path if os.path.isfile(path)]
        if not sample_candidates and available_val_mbin:
            sample_candidates = available_val_mbin[: args.inference_num_cases]
        else:
            sample_candidates = sample_candidates[: args.inference_num_cases]
        if not sample_candidates:
            print("❌ 未找到sample数据的.mbin文件")
            raise SystemExit(1)

        sample_data = MAPFDataset(
            sample_candidates,
            args.feature_dim,
            args.feature_type,
            first_step=True,
        )
        sample_loader = DataLoader(
            sample_data,
            shuffle=False,
            batch_size=1,
            num_workers=1,
        )

    online_schedule = get_online_schedule(args) if args.dataset_mode == "online" else None

    print_runtime_summary(
        args,
        net,
        device,
        train_loaders,
        train_loader_weights,
        val_loaders,
        sample_candidates,
        online_schedule,
    )
    train(
        args,
        net,
        train_loaders,
        train_loader_weights,
        val_loaders,
        sample_loader,
        optimizer,
        loss_fn,
        device,
    )
