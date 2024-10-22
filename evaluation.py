import torch

def evaluate_valid_loss(model, val_loader, loss_fn, device):
    """
    Evaluates the model on the validation set.

    Args:
        args: Argument object that contains configurations like learning rate and epochs.
        model: The neural network model (UNet).
        val_loader: Dataloader for the validation dataset.
        loss_fn: Loss function.
        device: Device to run the evaluation on (default is 'cuda').

    Returns:
        val_loss (float): The average validation loss for the entire validation set.
    """
    # Set model to evaluation mode
    model.eval()  
    val_loss = 0
    with torch.no_grad():  # Disable gradient calculation
        for batch in val_loader:
            # Load validation data onto the correct device (CPU/GPU)
            feature = batch["feature"].to(device)
            action_y = batch["action"].to(device)
            mask = batch["mask"].to(device)

            # Forward pass
            logits, _ = model(feature)

            # Compute the loss and apply mask
            loss = loss_fn(logits, action_y)
            loss = loss * mask.float()
            val_loss += loss.sum().item()

    val_loss /= len(val_loader)  
    return val_loss