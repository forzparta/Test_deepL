import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


def setup_training_components(model, learning_rate=0.01, weight_decay=0.01, momentum = 0, scheduler_type = None,  step_size=30, gamma=0.3, optimizer_type='SGD'):
    """
    Setup the criterion, optimizer, and scheduler for training.

    Args:
    model (torch.nn.Module): The neural network model.
    learning_rate (float): Learning rate for the optimizer.
    weight_decay (float): Weight decay for the optimizer.
    momentum: SDG momentum
    scheduler_type(str): Type of scheduler ('StepLR').
    step_size (int): Step size for the learning rate scheduler.
    gamma (float): Multiplicative factor of learning rate decay.
    optimizer_type (str): Type of optimizer ('SGD').

    Returns:
    criterion, optimizer, scheduler
    """
    #needs to be separed in future
    criterion = nn.CrossEntropyLoss()

    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum= momentum)
    else:
        raise ValueError("Unsupported optimizer type. Choose 'SGD'.")

    scheduler_f = None
    if scheduler_type == "StepLR":
        scheduler_f = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    return criterion, optimizer, scheduler_f