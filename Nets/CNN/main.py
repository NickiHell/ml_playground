import glob
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Tuple

import pytz
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from PIL import Image
from loguru import logger
from matplotlib import pyplot as plt
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FILE_DIR = Path(__file__).resolve().parent


class CNNNet(nn.Module):
    """Более крутая штука для классификации рыбок и котов"""

    # TODO: Разобраться че за хуйня с обучением

    def __init__(self, epoch=10, batch_size=128, num_classes=2):
        super(CNNNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(6, 6), stride=(2, 2), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2),
            nn.Conv2d(64, 192, kernel_size=(5, 5), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_classes)
        )

        self.batch_size = batch_size
        self.epoch = epoch
        self.train_data_path = f'{BASE_DIR}/Datasets/Classification/CatsDogs/training/'
        self.val_data_path = f'{BASE_DIR}/Datasets/Classification/CatsDogs/validation/'

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.optimizer = Adam(self.parameters(), lr=0.001)
        self.loss_fn = CrossEntropyLoss()
        self.to(self.device)

        self.accuracy = []

        logger.info(f'Device: {self.device.type}')

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def predict(self, path: str, who: str) -> bool:
        labels = ['cat', 'dog']

        img: Image = Image.open(path)
        img = self._image_transforms()(img).to(self.device)
        img = torch.unsqueeze(img, 0)

        self.eval()
        prediction = F.softmax(self(img), dim=1)
        prediction = prediction.argmax()

        try:
            assert labels[prediction] == who
        except AssertionError:
            f'Prediction Error: {labels[prediction]} != {who}'
            return False
        return True

    def save_model(self, path: str = f'{FILE_DIR}/result/cnn_net') -> None:
        torch.save(self.state_dict(), path)

    def load_model(self, path: str = f'{FILE_DIR}/result/cnn_net') -> None:
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

    def train_net(self):
        training_loader, validation_loader = self._collect_loaders()
        for epoch in range(1, self.epoch + 1):
            training_loss = 0.0
            valid_loss = 0.0
            self.train()
            for batch in training_loader:
                self.optimizer.zero_grad()
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                output = self(inputs)
                loss = self.loss_fn(output, targets)
                loss.backward()
                self.optimizer.step()
                training_loss += loss.data.item() * inputs.size(0)
            training_loss /= len(training_loader.dataset)

            self.eval()
            num_correct = 0
            num_examples = 0
            for batch in validation_loader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                output = self(inputs)
                targets = targets.to(self.device)
                loss = self.loss_fn(output, targets)
                valid_loss += loss.data.item() * inputs.size(0)
                correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1],
                                   targets)
                num_correct += torch.sum(correct).item()
                num_examples += correct.shape[0]
            valid_loss /= len(validation_loader.dataset)
            self.accuracy.append(num_correct / num_examples)
            logger.info(f'Epoch: {epoch}, Training Loss: {training_loss:.2f}, '
                        f'Validation Loss: {valid_loss:.2f}, accuracy = {num_correct / num_examples:.2f}')

    def _collect_data(self) -> Tuple[ImageFolder, ...]:
        dataset = partial(ImageFolder)
        image_transforms = self._image_transforms()
        train_data = dataset(root=self.train_data_path, transform=image_transforms)
        val_data = dataset(root=self.val_data_path, transform=image_transforms)
        return train_data, val_data

    def _collect_loaders(self) -> Tuple[DataLoader, ...]:
        loader = partial(DataLoader)
        data: Tuple[ImageFolder, ...] = self._collect_data()
        train_data_loader = loader(data[0], batch_size=self.batch_size)
        val_data_loader = loader(data[1], batch_size=self.batch_size)
        return train_data_loader, val_data_loader

    @staticmethod
    def _image_transforms() -> Compose:
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


if __name__ == '__main__':
    cnn_net = CNNNet(epoch=5, batch_size=128)
    # cnn_net.load_model()
    cnn_net.train_net()
    cnn_net.show_plot()
    cnn_net.save_plot()
    cnn_net.save_model()

    cats = glob.glob(f'{BASE_DIR}/Datasets/Classification/CatsDogs/validation/cats/*')
    dogs = glob.glob(f'{BASE_DIR}/Datasets/Classification/CatsDogs/validation/dogs/*')

    succeed_cats = [cnn_net.predict(x, 'cat') for x in cats if cnn_net.predict(x, 'cat')]
    succeed_dogs = [cnn_net.predict(x, 'dog') for x in dogs if cnn_net.predict(x, 'dog')]

    logger.info(f'Succeed Cats: {len(succeed_cats)}/{len(cats)}')
    logger.info(f'Succeed Dogs: {len(succeed_dogs)}/{len(dogs)}')
