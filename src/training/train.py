import os
import torch
import time
# import aim  # https://aimstack.io/#demos
import pandas as pd
from tqdm import tqdm
from log.logger import setup_logger
from omegaconf import OmegaConf

from utils import save_checkpoint, mkdir, check_path_exists, get_device
from models.get_model import get_model
from data_handling.data_loader import get_data_loaders
from training.optimizer import get_optimizer


def train_fold(config: OmegaConf, fold: int = 0):
    logger = setup_logger()

    # Initialize Aim run
    # aim_run = aim.Run()
    # aim_run.set_params(config.to_dict(), name='config')

    # gpu selection
    device = get_device(config, logger)
    # device = "cpu"  # FIXME for development

    # Create checkpoint directory
    checkpoint_path = os.path.join(os.environ["PROJECT_PATH"], os.environ["SUBPROJECT"], "weights")
    mkdir(checkpoint_path)
    logger.info(f"Checkpoint path: {checkpoint_path}")

    model = get_model(config, device, logger)
    optimizer = get_optimizer(model, config.optimizer)

    criterion = torch.nn.MSELoss()  # Define your loss function
    train_loader, val_loader = get_data_loaders(config, fold=fold)

    losses = {}

    logger.info("Starting training")
    start_time = time.time()
    for epoch in range(1, config.epochs + 1):
        # Training
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            # data and target are lists
            # data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze().float(), target.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        losses[epoch] = {"epoch": epoch, "train_loss": train_loss}
        logger.info(f'Epoch {epoch}, Loss: {train_loss}')

        # Log training loss to Aim
        # aim_run.track(train_loss, name='train_loss', epoch=epoch)

        # Validation
        if ((epoch % config.val_freq == 0 or epoch % config.save_freq == 0 or epoch == config.epochs)
                and epoch >= config.warmup):  # ensure validation if stored
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data, target in val_loader:
                    # data and target are lists
                    # data = data.to(device)
                    target = target.to(device)
                    output = model(data)
                    output, target = output.squeeze().float(), target.float()
                    loss = criterion(output, target)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            losses[epoch].update({"val_loss": val_loss})
            logger.info(f'Validation loss: {val_loss}')

            # Log validation loss to Aim
            # aim_run.track(val_loss, name='val_loss', epoch=epoch)

        # Save checkpoint
        if (epoch % config.save_freq == 0 and epoch >= config.warmup) or epoch == config.epochs:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=os.path.join(checkpoint_path, f'checkpoint_{epoch}_fold-{fold}.pth.tar'))
            losses[epoch].update({"stored": 1})
            logger.info(f'Checkpoint saved at epoch {epoch}')

    end_time = time.time()

    # Save losses to a CSV file
    pd.DataFrame(losses).T.to_csv(os.path.join(checkpoint_path, f"losses_fold-{fold}.csv"))
    logger.info(f"Training process completed. Training time: {round((end_time - start_time)/60, 4)} mins.")
    logger.info(f"Weights path: {checkpoint_path}")


def train(config: OmegaConf):
    # TODO possibility of parallelization across folds!
    for fold in range(config.nr_folds):
        train_fold(config, fold)
        # train_fold(config, 1)  # for development
