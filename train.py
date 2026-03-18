import glob
import inspect
import os
import random
import time
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


class NullSummaryWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def add_text(self, *args, **kwargs):
        return None

    def add_video(self, *args, **kwargs):
        return None

    def flush(self):
        return None

    def close(self):
        return None


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
            **make_dataloader_kwargs(
                args.batch_size,
                args.num_workers,
                sampler=val_sampler,
            ),
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
            **make_dataloader_kwargs(
                args.batch_size,
                args.num_workers,
                sampler=train_sampler,
            ),
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
            **make_dataloader_kwargs(
                args.batch_size,
                args.num_workers,
            ),
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


def format_args_text(args):
    rows = []
    for key in sorted(vars(args)):
        if key == "writer":
            continue
        rows.append(f"{key}: {getattr(args, key)}")
    return "\n".join(rows)


def estimate_total_train_steps(args, train_loaders):
    if args.dataset_mode == "online":
        return int(args.online_total_steps)
    return int(sum(len(loader) for loader in train_loaders.values()) * args.epochs)


def setup_run_logging(args, total_train_steps):
    args.estimated_total_train_steps = total_train_steps
    args.tensorboard_enabled = total_train_steps >= 1000
    args.writer = NullSummaryWriter()

    if args.local_rank != 0:
        return

    os.makedirs(args.real_log_dir, exist_ok=True)
    args_text = format_args_text(args)
    print(args_text)

    if not args.tensorboard_enabled:
        print(
            f"TensorBoard disabled: estimated_total_train_steps={total_train_steps} < 1000"
        )
        return

    args.writer = SummaryWriter(log_dir=args.real_log_dir)
    args.writer.add_text("Args", f"```text\n{args_text}\n```", 0)


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


def make_dataloader_kwargs(batch_size, num_workers, *, sampler=None, shuffle=False):
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "shuffle": shuffle,
    }
    if sampler is not None:
        kwargs["sampler"] = sampler
        kwargs.pop("shuffle", None)
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    if torch.cuda.is_available():
        kwargs["pin_memory"] = True
    return kwargs


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
        ("config_source", getattr(args, "config_source", "CLI only")),
        ("run_timestamp", args.current_time),
        ("run_dir", args.real_log_dir),
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
        ("estimated_total_train_steps", getattr(args, "estimated_total_train_steps", "unknown")),
        ("tensorboard_enabled", getattr(args, "tensorboard_enabled", "unknown")),
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
        all_paths, all_goal_locations, _, file_name, metrics = path_formation(
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

        _log_inference_video(args, sample_loader, idx, all_paths, all_goal_locations, log_step)

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


def _masked_entropy(logit, mask):
    with torch.no_grad():
        probs = torch.softmax(logit.detach(), dim=1)
        log_probs = torch.log_softmax(logit.detach(), dim=1)
        entropy_map = -(probs * log_probs).sum(dim=1)
        mask_float = mask.detach().float()
        valid = mask_float.sum().item()
        if valid <= 0:
            return float("nan")
        entropy_value = (entropy_map * mask_float).sum().item() / valid
    return float(entropy_value)


def _gpu_memory_mb(device):
    if device.type != "cuda" or not torch.cuda.is_available():
        return None, None, None, None
    allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
    max_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    max_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    return float(allocated_mb), float(reserved_mb), float(max_allocated_mb), float(max_reserved_mb)


def _reset_gpu_peak_memory(device):
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def _sync_device(device):
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _log_train_step_metrics(
    args,
    optimizer,
    dims,
    log_step,
    step_loss,
    data_fetch_sec,
    step_time_sec,
    entropy_value,
    device,
):
    if args.local_rank != 0:
        return

    dims_tag = f"{dims[0]}x{dims[1]}"
    args.writer.add_scalar("Optimization/LR", optimizer.param_groups[0]["lr"], log_step)
    args.writer.add_scalar("Loss/TrainStep", step_loss, log_step)
    args.writer.add_scalar(f"Loss/TrainStep_{dims_tag}", step_loss, log_step)
    args.writer.add_scalar("Time/DataFetch_s", data_fetch_sec, log_step)
    args.writer.add_scalar("Time/Step_s", step_time_sec, log_step)

    if np.isfinite(entropy_value):
        args.writer.add_scalar("Entropy/Train", entropy_value, log_step)

    allocated_mb, reserved_mb, max_allocated_mb, max_reserved_mb = _gpu_memory_mb(device)
    if allocated_mb is not None:
        args.writer.add_scalar("GPU/memory_allocated_mb", allocated_mb, log_step)
        args.writer.add_scalar("GPU/memory_reserved_mb", reserved_mb, log_step)
        args.writer.add_scalar("GPU/max_memory_allocated_mb", max_allocated_mb, log_step)
        args.writer.add_scalar("GPU/max_memory_reserved_mb", max_reserved_mb, log_step)


def _downsample_video_indices(num_frames, max_frames):
    if num_frames <= max_frames:
        return np.arange(num_frames, dtype=np.int64)
    return np.linspace(0, num_frames - 1, num=max_frames, dtype=np.int64)


def _render_inference_video(sample_data, all_paths, all_goal_locations, max_frames=128, cell_size=8):
    map_data = sample_data["feature"][0].detach().cpu().numpy()
    height, width = map_data.shape

    base = np.full((height, width, 3), 255, dtype=np.uint8)
    obstacle_mask = map_data > 0
    base[obstacle_mask] = np.array([32, 32, 32], dtype=np.uint8)

    trail_counts = np.zeros((height, width), dtype=np.int32)
    indices = _downsample_video_indices(len(all_paths), max_frames)
    frames = []

    for frame_idx in indices:
        frame = base.copy()

        path_positions = np.asarray(all_paths[frame_idx], dtype=np.int64)
        goal_positions = np.asarray(all_goal_locations[frame_idx], dtype=np.int64)

        for row, col in path_positions:
            if 0 <= row < height and 0 <= col < width:
                trail_counts[row, col] += 1

        trail_mask = trail_counts > 0
        frame[trail_mask & ~obstacle_mask] = np.array([170, 210, 255], dtype=np.uint8)

        for row, col in goal_positions:
            if 0 <= row < height and 0 <= col < width and not obstacle_mask[row, col]:
                frame[row, col] = np.array([80, 200, 120], dtype=np.uint8)

        for row, col in path_positions:
            if 0 <= row < height and 0 <= col < width and not obstacle_mask[row, col]:
                frame[row, col] = np.array([220, 80, 80], dtype=np.uint8)

        overlap_mask = np.zeros((height, width), dtype=bool)
        for row, col in path_positions:
            if 0 <= row < height and 0 <= col < width:
                overlap_mask[row, col] = True
        for row, col in goal_positions:
            if 0 <= row < height and 0 <= col < width and overlap_mask[row, col]:
                frame[row, col] = np.array([255, 210, 70], dtype=np.uint8)

        if cell_size > 1:
            frame = np.repeat(np.repeat(frame, cell_size, axis=0), cell_size, axis=1)

        frames.append(frame.transpose(2, 0, 1))

    video = np.stack(frames, axis=0)
    return torch.from_numpy(video).unsqueeze(0)


def _log_inference_video(args, sample_loader, sample_idx, all_paths, all_goal_locations, log_step):
    if args.local_rank != 0 or sample_idx != 0 or not hasattr(args.writer, "add_video"):
        return

    sample_data = sample_loader.dataset[sample_idx]
    video = _render_inference_video(sample_data, all_paths, all_goal_locations)
    args.writer.add_video("InferenceVideo/case_0", video, log_step, fps=4)


def train_offline(args, model, train_loaders, val_loaders, sample_loader, optimizer, loss_fn, device):
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        total_agents = 0
        total_fetch_sec = 0.0
        total_step_sec = 0.0
        total_entropy = 0.0
        entropy_count = 0
        step_count = 0
        dim_loss_sums = {dims: 0.0 for dims in train_loaders.keys()}
        dim_agent_sums = {dims: 0 for dims in train_loaders.keys()}

        for dims, train_loader in train_loaders.items():
            sampler = getattr(train_loader, "sampler", None)
            if args.distributed and sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            train_iter = iter(train_loader)
            for _ in tqdm(
                range(len(train_loader)),
                desc=f"Epoch {epoch}/{args.epochs}",
                disable=args.local_rank != 0,
                total=len(train_loader),
            ):
                fetch_start = time.perf_counter()
                batch = next(train_iter)
                data_fetch_sec = time.perf_counter() - fetch_start

                _sync_device(device)
                _reset_gpu_peak_memory(device)
                step_start = time.perf_counter()
                feature = batch["feature"].to(device)
                action_y = batch["action"].to(device)
                mask = batch["mask"].to(device)

                logit, _ = model(feature)
                loss = loss_fn(logit, action_y)
                masked_loss = loss * mask.float()
                mask_sum = mask.detach().sum().item()
                if mask_sum <= 0:
                    continue
                averaged_loss = masked_loss.sum() / mask_sum
                entropy_value = _masked_entropy(logit, mask)

                optimizer.zero_grad()
                averaged_loss.backward()
                optimizer.step()
                _sync_device(device)
                step_time_sec = time.perf_counter() - step_start

                train_loss += masked_loss.detach().sum().item()
                total_agents += mask_sum
                total_fetch_sec += data_fetch_sec
                total_step_sec += step_time_sec
                if np.isfinite(entropy_value):
                    total_entropy += entropy_value
                    entropy_count += 1
                step_count += 1
                dim_loss_sums[dims] += masked_loss.detach().sum().item()
                dim_agent_sums[dims] += mask_sum

                _log_train_step_metrics(
                    args=args,
                    optimizer=optimizer,
                    dims=dims,
                    log_step=global_step,
                    step_loss=float(averaged_loss.detach().item()),
                    data_fetch_sec=data_fetch_sec,
                    step_time_sec=step_time_sec,
                    entropy_value=entropy_value,
                    device=device,
                )
                global_step += 1

        train_loss = train_loss / total_agents if total_agents > 0 else float("inf")
        if args.local_rank == 0:
            args.writer.add_scalar("Loss/Train", train_loss, epoch)
            for dims in sorted(dim_loss_sums.keys()):
                dim_agents = dim_agent_sums[dims]
                if dim_agents <= 0:
                    continue
                dims_tag = f"{dims[0]}x{dims[1]}"
                args.writer.add_scalar(f"Loss/Train_{dims_tag}", dim_loss_sums[dims] / dim_agents, epoch)
            if step_count > 0:
                args.writer.add_scalar("Time/DataFetch_s_epoch_avg", total_fetch_sec / step_count, epoch)
                args.writer.add_scalar("Time/Step_s_epoch_avg", total_step_sec / step_count, epoch)
            if entropy_count > 0:
                args.writer.add_scalar("Entropy/Train_epoch_avg", total_entropy / entropy_count, epoch)
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
    fetch_time_window = 0.0
    step_time_window = 0.0
    entropy_sum_window = 0.0
    entropy_count_window = 0
    window_step_count = 0
    dim_loss_window = {dims: 0.0 for dims in dims_list}
    dim_agents_window = {dims: 0 for dims in dims_list}
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
        fetch_start = time.perf_counter()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            train_iters[dims] = train_iter
            batch = next(train_iter)
        data_fetch_sec = time.perf_counter() - fetch_start

        _sync_device(device)
        _reset_gpu_peak_memory(device)
        step_start = time.perf_counter()
        feature = batch["feature"].to(device)
        action_y = batch["action"].to(device)
        mask = batch["mask"].to(device)

        logit, _ = model(feature)
        loss = loss_fn(logit, action_y)
        masked_loss = loss * mask.float()
        mask_sum = mask.detach().sum().item()
        if mask_sum <= 0:
            continue
        averaged_loss = masked_loss.sum() / mask_sum
        entropy_value = _masked_entropy(logit, mask)

        optimizer.zero_grad()
        averaged_loss.backward()
        optimizer.step()
        _sync_device(device)
        step_time_sec = time.perf_counter() - step_start

        train_loss_window += masked_loss.detach().sum().item()
        total_agents_window += mask_sum
        fetch_time_window += data_fetch_sec
        step_time_window += step_time_sec
        window_step_count += 1
        if np.isfinite(entropy_value):
            entropy_sum_window += entropy_value
            entropy_count_window += 1
        dim_loss_window[dims] += masked_loss.detach().sum().item()
        dim_agents_window[dims] += mask_sum

        _log_train_step_metrics(
            args=args,
            optimizer=optimizer,
            dims=dims,
            log_step=step,
            step_loss=float(averaged_loss.detach().item()),
            data_fetch_sec=data_fetch_sec,
            step_time_sec=step_time_sec,
            entropy_value=entropy_value,
            device=device,
        )
        if args.local_rank == 0:
            args.writer.add_scalar("Loss/Train", float(averaged_loss.detach().item()), step)

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
            args.writer.add_scalar("Loss/Train_window_avg", train_loss, step)
            for dims_key in dims_list:
                dim_agents = dim_agents_window[dims_key]
                if dim_agents <= 0:
                    continue
                dims_tag = f"{dims_key[0]}x{dims_key[1]}"
                args.writer.add_scalar(f"Loss/Train_{dims_tag}", dim_loss_window[dims_key] / dim_agents, step)
            if window_step_count > 0:
                args.writer.add_scalar("Time/DataFetch_s_window_avg", fetch_time_window / window_step_count, step)
                args.writer.add_scalar("Time/Step_s_window_avg", step_time_window / window_step_count, step)
            if entropy_count_window > 0:
                args.writer.add_scalar("Entropy/Train_window_avg", entropy_sum_window / entropy_count_window, step)
            print(f"Step {step}/{total_steps}, Training mean Loss: {train_loss}")
            train_loss_window = 0.0
            total_agents_window = 0
            fetch_time_window = 0.0
            step_time_window = 0.0
            entropy_sum_window = 0.0
            entropy_count_window = 0
            window_step_count = 0
            dim_loss_window = {dims_key: 0.0 for dims_key in dims_list}
            dim_agents_window = {dims_key: 0 for dims_key in dims_list}

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
    args.current_time = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    args.real_log_dir = os.path.join(args.log_dir, f"{args.current_time}")
    args.writer = NullSummaryWriter()

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
            **make_dataloader_kwargs(
                1,
                1,
                shuffle=False,
            ),
        )

    online_schedule = get_online_schedule(args) if args.dataset_mode == "online" else None
    estimated_total_train_steps = estimate_total_train_steps(args, train_loaders)
    setup_run_logging(args, estimated_total_train_steps)

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
    args.writer.flush()
    args.writer.close()
