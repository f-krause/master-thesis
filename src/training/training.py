import os
import torch
import pandas as pd
from tqdm import tqdm
from data_handling.data_loader import get_data_loaders
from training.optimizer import get_optimizer
from models.model_template import ModelBaseline
from utils import save_checkpoint, mkdir, check_path_exists
from log.logger import setup_logger


def train(config):
    logger = setup_logger()
    logger.info("Starting training")

    checkpoint_path = os.path.join(os.environ["PROJECT_PATH"], os.environ["SUBPROJECT"], "weights")
    checkpoint_path = check_path_exists(checkpoint_path, create_unique=True)
    mkdir(checkpoint_path)
    logger.info(f"Checkpoint path: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = ModelBaseline().to(device)  # Initialize your model
    optimizer = get_optimizer(model, config)

    criterion = torch.nn.MSELoss()  # Define your loss function
    train_loader, val_loader = get_data_loaders(config['batch_size'], config['num_workers'])

    losses = {}

    for epoch in range(1, config['epochs'] + 1):
        # Training
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        losses[epoch] = {"train": running_loss / len(train_loader)}
        logger.info(f'Epoch {epoch}, Loss: {running_loss / len(train_loader)}')

        # Validation
        if (epoch % config['val_freq'] == 0 and epoch > config["warmup"]) or epoch == config['epochs'] - 1:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    outputs = model(data)
                    loss = criterion(outputs, target)
                    val_loss += loss.item()
            losses[epoch].update({"val": val_loss / len(val_loader)})
            logger.info(f'Validation loss: {val_loss / len(val_loader)}')

        # Save checkpoint
        if (epoch % config['save_freq'] == 0 and epoch > config["warmup"]) or epoch == config['epochs'] - 1:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=os.path.join(checkpoint_path, f'checkpoint_{epoch}.pth.tar'))
            losses[epoch].update({"stored": 1})
            logger.info(f'Checkpoint saved at epoch {epoch}')

    pd.DataFrame(losses).T.to_csv(os.path.join(checkpoint_path, "losses.csv"))
    logger.info("Training process completed")
