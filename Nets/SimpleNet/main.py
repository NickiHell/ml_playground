import torchvision
from torch import nn
from torch.autograd.grad_mode import F
from torch.utils import data
from torchvision import transforms

if __name__ == '__main__':
    train_data_path = './train/'
    val_data_path = './val/'
    test_data_path = './test/'
    batch_size = 64

    transforms = transforms.Compose([
        transforms.Resize(64),
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
            self.fc1 = nn.Linear(12228, 84)
            self.fc2 = nn.Linear(84, 50)
            self.fc3 = nn.Linear(50, 2)

        def forward(self, x):
            x = x.view(-1, 12228)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    simplenet = SimpleNet()
