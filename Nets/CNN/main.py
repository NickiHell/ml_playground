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

    # TODO: Глупее со временем, починить

    def __init__(self, epoch=10, batch_size=128, num_classes=2):
        super(CNNNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(6, 6), stride=(4, 4), padding=2),
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
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
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
        self.plot = plt.plot()

        logger.info(f'Device: {self.device.type}')

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def predict(self, path: str) -> bool:
        labels = ['cat', 'dog']
        result = 'cat' if 'cat' in path else 'dog'

        img: Image = Image.open(path)
        img = self._image_transforms()(img).to(self.device)
        img = torch.unsqueeze(img, 0)

        self.eval()
        prediction = F.softmax(self(img), dim=1)
        prediction = prediction.argmax()

        try:
            assert labels[prediction] == result
        except AssertionError:
            f'Prediction Error: {labels[prediction]} != {result}'
            return False
        return True

    def save_model(self, path: str = f'{FILE_DIR}/result/cnn_net') -> None:
        torch.save(self.state_dict(), path)

    def load_model(self, path: str = f'{FILE_DIR}/result/cnn_net') -> None:
        self.load_state_dict(torch.load(path))

    def show_plot(self) -> None:
        self.plot = plt.plot([x for x in range(1, self.epoch + 1)], [x for x in self.accuracy], 'r--')
        self.plot.xlabel('Epoch')
        self.plot.ylabel('Accuracy')
        self.plot.show()

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

    def find_lr(self, init_value=1e-8, final_value=10.0):
        train_loader, _ = self._collect_loaders()
        number_in_epoch = len(train_loader) - 1
        update_step = (final_value / init_value) ** (1 / number_in_epoch)
        lr = init_value
        self.optimizer.param_groups[0]["lr"] = lr
        best_loss = 0.0
        batch_num = 0
        losses = []
        log_lrs = []
        for data in train_loader:
            batch_num += 1
            inputs, targets = data
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.loss_fn(outputs, targets)

            # Crash out if loss explodes

            if batch_num > 1 and loss > 4 * best_loss:
                if (len(log_lrs) > 20):
                    return log_lrs[10:-5], losses[10:-5]
                else:
                    return log_lrs, losses

            # Record the best loss

            if loss < best_loss or batch_num == 1:
                best_loss = loss

            # Store the values
            losses.append(loss.item())
            log_lrs.append((lr))

            # Do the backward pass and optimize

            loss.backward()
            self.optimizer.step()

            # Update the lr for the next step and store

            lr *= update_step
            self.optimizer.param_groups[0]["lr"] = lr
        logger.warning(f'LR Found: {lr:2f}')

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
    cnn_net = CNNNet(epoch=25, batch_size=64)
    # cnn_net.load_model()
    cnn_net.train_net()
    cnn_net.find_lr()
    cnn_net.show_plot()
    cnn_net.save_model()

    cats = glob.glob(f'{BASE_DIR}/Datasets/Classification/CatsDogs/validation/cats/*')
    dogs = glob.glob(f'{BASE_DIR}/Datasets/Classification/CatsDogs/validation/dogs/*')

    logger.info(f'Cats errors: {len([cnn_net.predict(x) for x in cats if cnn_net.predict(x)]) / len(cats)}')
    logger.info(f'Dogs errors: {len([cnn_net.predict(x) for x in dogs if cnn_net.predict(x)]) / len(dogs)}')
