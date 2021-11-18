import torch
import torch.nn.functional as F
import torchvision
from torch import nn, optim
from torch.utils import data
from torchvision import transforms

if __name__ == '__main__':
    train_data_path = 'dataset/train/'
    val_data_path = 'dataset/val/'
    test_data_path = 'dataset/test/'
    batch_size = 1

    transforms = transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.255],
        )
    ])

    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=transforms)
    val_data = torchvision.datasets.ImageFolder(root=val_data_path, transform=transforms)
    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=transforms)

    train_data_loader = data.DataLoader(train_data, batch_size=batch_size)
    val_data_loader = data.DataLoader(val_data, batch_size=batch_size)
    test_data_loader = data.DataLoader(test_data, batch_size=batch_size)


    class SimpleNet(nn.Module):
        """Простая нейронка которая отличает кошек от собак"""

        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(12288, 84)
            self.fc2 = nn.Linear(84, 50)
            self.fc3 = nn.Linear(50, 2)

        def forward(self, x):
            x = x.view(-1, 12288)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    simplenet = SimpleNet()
    optimizer = optim.Adam(simplenet.parameters(), lr=0.001)


    def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device="cpu"):
        for epoch in range(1, epochs + 1):
            training_loss = 0.0
            valid_loss = 0.0
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                output = model(inputs)
                loss = loss_fn(output, targets)
                loss.backward()
                optimizer.step()
                training_loss += loss.data.item() * inputs.size(0)
            training_loss /= len(train_loader.dataset)

            model.eval()
            num_correct = 0
            num_examples = 0
            for batch in val_loader:
                inputs, targets = batch
                inputs = inputs.to(device)
                output = model(inputs)
                targets = targets.to(device)
                loss = loss_fn(output, targets)
                valid_loss += loss.data.item() * inputs.size(0)
                correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
                num_correct += torch.sum(correct).item()
                num_examples += correct.shape[0]
            valid_loss /= len(val_loader.dataset)

            print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.format(epoch,
                                                                                                        training_loss,
                                                                                                        valid_loss,
                                                                                                        num_correct / num_examples))


    train(simplenet, optimizer, torch.nn.CrossEntropyLoss(), train_data_loader, val_data_loader, epochs=5)
