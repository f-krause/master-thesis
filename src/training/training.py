import torch
from tqdm import tqdm
from ..data.data_loader import get_data_loaders
from ..optimization.optimizer import get_optimizer
from ..models.model_template import ModelBaseline
from ..training.utils import save_checkpoint
from ..logs.logger import setup_logger


def train(config):
    logger = setup_logger()
    logger.info("Starting training process")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = ModelBaseline().to(device)  # Initialize your model
    optimizer = get_optimizer(model, config)

    criterion = torch.nn.CrossEntropyLoss()
    train_loader, val_loader = get_data_loaders(config['batch_size'], config['num_workers'])

    for epoch in range(config['epochs']):
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

        logger.info(f'Epoch {epoch}, Loss: {running_loss / len(train_loader)}')

        if epoch % config['save_freq'] == 0:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=f'checkpoint_{epoch}.pth.tar')
            logger.info(f'Checkpoint saved at epoch {epoch}')

    logger.info("Training process completed")
