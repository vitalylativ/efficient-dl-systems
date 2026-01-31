import torch
from torch import nn
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from tqdm.auto import tqdm

from unet import Unet

from dataset import get_train_data


def _check_overflow(model: torch.nn.Module) -> bool:
    for param in model.parameters():
        if param.grad is not None:
            if not torch.isfinite(param.grad).all():
                return True
    return False


def _unscale_grads(model: torch.nn.Module, loss_scale: float) -> None:
    for param in model.parameters():
        if param.grad is None:
            continue
        param.grad.mul_(1.0 / loss_scale)


def train_epoch(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mode: str = "static",
    scaler: GradScaler | None = None,
    loss_scale_state: dict | None = None,
) -> None:
    model.train()

    if mode == "static" and loss_scale_state is None:
        loss_scale_state = {"loss_scale": 2.0**16}
    elif mode == "dynamic" and loss_scale_state is None:
        loss_scale_state = {
            "loss_scale": 2.0**16,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
            "successful_steps": 0,
        }

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if mode == "debug":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        elif mode == "static":
            loss_scale = loss_scale_state["loss_scale"]
            (loss * loss_scale).backward()

            if _check_overflow(model):
                optimizer.zero_grad()
                continue

            _unscale_grads(model, loss_scale)
            optimizer.step()

        elif mode == "dynamic":
            loss_scale = loss_scale_state["loss_scale"]
            (loss * loss_scale).backward()

            if _check_overflow(model):
                optimizer.zero_grad()
                loss_scale_state["loss_scale"] *= loss_scale_state["backoff_factor"]
                loss_scale_state["successful_steps"] = 0
                continue

            _unscale_grads(model, loss_scale)
            optimizer.step()

            loss_scale_state["successful_steps"] += 1
            if loss_scale_state["successful_steps"] >= loss_scale_state["growth_interval"]:
                loss_scale_state["loss_scale"] *= loss_scale_state["growth_factor"]
                loss_scale_state["successful_steps"] = 0

        else:
            raise ValueError(f"Unknown mode: {mode}")

        accuracy = ((outputs > 0.5) == labels).float().mean()

        pbar.set_description(f"Loss: {round(loss.item(), 4)} " f"Accuracy: {round(accuracy.item() * 100, 4)}")


def train(mode: str = "dynamic"):
    device = torch.device("cuda:0")
    model = Unet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loader = get_train_data()

    scaler = GradScaler() if mode == "debug" else None
    loss_scale_state = None

    num_epochs = 5
    for epoch in range(0, num_epochs):
        train_epoch(
            train_loader, model, criterion, optimizer,
            device=device, mode=mode, scaler=scaler,
            loss_scale_state=loss_scale_state,
        )
