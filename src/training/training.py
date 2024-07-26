import torch
from tqdm import tqdm
from utils import save_checkpoint
from ..data.data_loader import get_data_loaders
from ..optimization.optimizer import get_optimizer
# from ..models.model_a import ModelA  # TODO


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ModelA().to(device)  # TODO
    optimizer = get_optimizer(model, config)

    criterion = torch.nn.CrossEntropyLoss()
    train_loader, val_loader = get_data_loaders(config['batch_size'], config['num_workers'])

    for epoch in range(config['epochs']):
        model.train()
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch}, Loss: {loss.item()}')

        if epoch % config['save_freq'] == 0:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=f'checkpoint_{epoch}.pth.tar')
