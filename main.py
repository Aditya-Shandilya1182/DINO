import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from model import VisionTransformer
from dino_loss import DINOLoss
from train import train_dino, test_model
from utils import update_teacher

transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

student = VisionTransformer().cuda()
teacher = VisionTransformer().cuda()

for param in teacher.parameters():
    param.requires_grad = False

dino_loss = DINOLoss(out_dim=10).cuda()
optimizer = optim.Adam(student.parameters(), lr=1e-4)

num_epochs = 20
for epoch in range(num_epochs):
    train_dino(student, teacher, dino_loss, optimizer, train_loader, num_epochs=1)
    update_teacher(student, teacher)

test_model(student, test_loader)
