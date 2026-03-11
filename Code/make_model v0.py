# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 15:30:13 2025

@author: Adiepoer
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

# Pad to multiple of 16 (for UNet compatibility)
def pad_to_multiple(img, multiple=16):
    h, w = img.shape[:2]
    new_h = (h + multiple - 1) // multiple * multiple
    new_w = (w + multiple - 1) // multiple * multiple
    pad_h = new_h - h
    pad_w = new_w - w
    return cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

# Custom dataset
class EdgeDetectionDataset(Dataset):
    def __init__(self, image_dir, label_dir, image_transform=None, label_transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.image_files = os.listdir(image_dir)
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        # Pad to multiple of 16
        image = pad_to_multiple(image)
        label = pad_to_multiple(label)

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        label = label.astype(np.float32) / 255.0

        # Apply transforms
        if self.image_transform:
            image = self.image_transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1)

        if self.label_transform:
            label = self.label_transform(label)
        else:
            label = torch.from_numpy(label).unsqueeze(0)

        return image, label

# Transforms (no resizing)
image_transform = transforms.ToTensor()
label_transform = transforms.ToTensor()

# Load dataset
train_dataset = EdgeDetectionDataset(
    image_dir='D:/My Project/auto_trace/learning/tes_dataset/train/images',
    label_dir='D:/My Project/auto_trace/learning/tes_dataset/train/labels',
    image_transform=image_transform,
    label_transform=label_transform
)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Check sample
images, labels = next(iter(train_loader))
print("Image shape:", images.shape)
print("Label shape:", labels.shape)

# Visualize
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(images[0].permute(1, 2, 0).cpu().numpy())
plt.title('Image')
plt.subplot(122)
plt.imshow(labels[0][0].cpu().numpy(), cmap='gray')
plt.title('Label')
plt.show()

# Model
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class EdgeDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(3, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)

        self.maxpool = nn.MaxPool2d(2)

        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up_conv1 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_conv2 = DoubleConv(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up_conv3 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.maxpool(x1))
        x3 = self.down3(self.maxpool(x2))
        x4 = self.down4(self.maxpool(x3))

        x = self.up1(x4)
        x = self.up_conv1(torch.cat([x, x3], dim=1))

        x = self.up2(x)
        x = self.up_conv2(torch.cat([x, x2], dim=1))

        x = self.up3(x)
        x = self.up_conv3(torch.cat([x, x1], dim=1))

        return self.sigmoid(self.final_conv(x))

# Loss
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        if targets.size(1) == 3:
            targets = targets.mean(dim=1, keepdim=True)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        bce = F.binary_cross_entropy(inputs, targets)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + 1.) / (inputs.sum() + targets.sum() + 1.)
        return bce + (1 - dice)

# Train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EdgeDetectionModel().to(device)
criterion = DiceBCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(model, dataloader, criterion, optimizer, device, num_epochs=25):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}')

# Start training
train(model, train_loader, criterion, optimizer, device, num_epochs=25)

# Save model
torch.save(model.state_dict(), 'edge_detection_model.pth')


model = EdgeDetectionModel()
model.load_state_dict(torch.load('edge_detection_model.pth', map_location='cpu'))  # or 'cuda'
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def predict_edge(model, image_path, transform, device):
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = cv2.resize(image, (256, 256))  # Must match training size
    image = transform(image)
    image = image.unsqueeze(0).to(device)  # Add batch dimension

    # Predict
    with torch.no_grad():
        output = model(image)
        pred = output.squeeze().cpu().numpy()  # Remove batch and channel dims

    return pred

def overlay_edges_on_image(original_path, prediction, alpha=0.6):
    """
    original_path: path to the original image
    prediction: predicted edge map (2D numpy array, values between 0 and 1)
    alpha: blending weight
    """
    # Read and resize the original image (same size used in prediction)
    orig = cv2.imread(original_path)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    orig = cv2.resize(orig, (prediction.shape[1], prediction.shape[0]))

    # Convert prediction to uint8 and make into 3-channel red edge
    edges = (prediction * 255).astype(np.uint8)
    edges_color = np.zeros_like(orig)
    edges_color[:, :, 0] = edges  # Red channel

    # Blend edges onto original image
    overlayed = cv2.addWeighted(orig, 1.0, edges_color, alpha, 0)

    return overlayed


import matplotlib.pyplot as plt

img_path = 'D:/My Project/auto_trace/learning/tes_dataset/tes_model/FTIF_LTPMP-14-Aug-2020.png'
pred = predict_edge(model, img_path, image_transform, device)

overlayed_img = overlay_edges_on_image(img_path, pred)

plt.imshow(overlayed_img)
plt.title('Overlay of Edge Prediction')
plt.axis('off')
plt.show()
