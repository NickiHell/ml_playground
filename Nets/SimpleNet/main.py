import glob
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Tuple

import pytz
import torch
import torch.nn as nn
from PIL import Image
from loguru import logger
from matplotlib import pyplot as plt
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax, relu, selu
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FILE_DIR = Path(__file__).resolve().parent


class SimpleNet(nn.Module):
    """Простая нейронка которая отличает кошек от собак"""

    def __init__(self, batch_size: int = 64, epoch: int = 25) -> None:
        super().__init__()
        self.fc1 = nn.Linear(12288, 84)
        self.fc2 = nn.Linear(84, 50)
        self.fc3 = nn.Linear(50, 25)
        self.fc4 = nn.Linear(25, 2)

        self.batch_size = batch_size
        self.epoch = epoch
        self.train_data_path = f'{BASE_DIR}/Datasets/Classification/CatsDogs/training/'
        self.val_data_path = f'{BASE_DIR}/Datasets/Classification/CatsDogs/validation/'

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.optimizer = Adam(self.parameters(), lr=0.001)
        self.loss_fn = CrossEntropyLoss()
        self.to(self.device)

        logger.info(f'Device: {self.device.type}')

        self.accuracy = []

    def forward(self, x: Tensor) -> Tensor:
        x = x.reshape(-1, 12288)
        x = relu(self.fc1(x))
        x = selu(self.fc2(x))
        x = relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def predict(self, path: str) -> bool:
        labels = ['cat', 'dog']
        result = 'cat' if 'cat' in path else 'dog'

        img: Image = Image.open(path)
        img = self._image_transforms()(img).to(self.device)
        img = torch.unsqueeze(img, 0)

        self.eval()
        prediction = softmax(self(img), dim=1)
        prediction = prediction.argmax()

        try:
            assert labels[prediction] == result
        except AssertionError:
            f'Prediction Error: {labels[prediction]} != {result}'
            return False
        return True

    def save_model(self, path: str = f'{FILE_DIR}/result/simple_net') -> None:
        torch.save(self.state_dict(), path)

    def load_model(self, path: str = f'{FILE_DIR}/result/simple_net') -> None:
        self.load_state_dict(torch.load(path))

    def show_plot(self) -> None:
        plt.plot([x for x in range(1, self.epoch + 1)], [x for x in self.accuracy], 'r--')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.show()

    def save_plot(self) -> None:
        now = datetime.now(tz=pytz.utc)
        plt.savefig(
            f'{FILE_DIR}/plots/[{now.strftime("%H:%M:%S %d-%m-%Y")}] '
            f'Epochs:{self.epoch} Batch_size:{self.batch_size}.png')

    def train_net(self) -> None:
        training_data_loader, validation_data_loader = self._collect_loaders()
        for epoch in range(1, self.epoch + 1):
            training_loss = 0.0
            valid_loss = 0.0
            self.train()
            for batch in training_data_loader:
                self.optimizer.zero_grad()
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                output = self(inputs)
                loss = self.loss_fn(output, targets)
                loss.backward()
                self.optimizer.step()
                training_loss += loss.data.item() * inputs.size(0)
            training_loss /= len(training_data_loader.dataset)

            self.eval()
            num_correct = 0
            num_examples = 0
            for batch in validation_data_loader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                output = self(inputs)
                targets = targets.to(self.device)
                loss = self.loss_fn(output, targets)
                valid_loss += loss.data.item() * inputs.size(0)
                correct = torch.eq(torch.max(softmax(output, dim=1), dim=1)[1], targets)
                num_correct += torch.sum(correct).item()
                num_examples += correct.shape[0]
            valid_loss /= len(validation_data_loader.dataset)
            self.accuracy.append(num_correct / num_examples)
            logger.info(f'Epoch: {epoch}, Training Loss: {training_loss:.2f}, '
                        f'Validation Loss: {valid_loss:.2f}, accuracy = {num_correct / num_examples:.2f}')

    def _collect_data(self) -> Tuple[ImageFolder, ...]:
        dataset = partial(ImageFolder)
        image_transforms = self._image_transforms()
        training_data = dataset(root=self.train_data_path, transform=image_transforms)
        validation_data = dataset(root=self.val_data_path, transform=image_transforms)
        return training_data, validation_data

    def _collect_loaders(self) -> Tuple[DataLoader, ...]:
        loader = partial(DataLoader)
        data: Tuple[ImageFolder, ...] = self._collect_data()
        training_data_loader = loader(data[0], batch_size=self.batch_size)
        validation_data_loader = loader(data[1], batch_size=self.batch_size)
        return training_data_loader, validation_data_loader

    @staticmethod
    def _image_transforms() -> Compose:
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


if __name__ == '__main__':
    simple_net = SimpleNet(epoch=25, batch_size=512)
    # simple_net.load_model()
    simple_net.train_net()
    simple_net.show_plot()
    simple_net.save_plot()
    simple_net.save_model()

    cats = glob.glob(f'{BASE_DIR}/Datasets/Classification/CatsDogs/validation/cats/*')
    dogs = glob.glob(f'{BASE_DIR}/Datasets/Classification/CatsDogs/validation/dogs/*')

    logger.info(f'Cats errors: {len([simple_net.predict(x) for x in cats if simple_net.predict(x)]) / len(cats)}')
    logger.info(f'Dogs errors: {len([simple_net.predict(x) for x in dogs if simple_net.predict(x)]) / len(dogs)}')
