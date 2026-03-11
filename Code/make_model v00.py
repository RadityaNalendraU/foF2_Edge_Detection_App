# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 15:30:13 2025

@author: Adiepoer
"""

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

# ========================
# Dataset
# ========================
class EdgeDetectionDataset(Dataset):
    def __init__(self, image_dir, label_dir, image_transform=None, label_transform=None, target_size=(256, 256)):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.image_files = os.listdir(image_dir)
        self.target_size = target_size
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.image_files[idx])
        
        # Read and resize images
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.target_size)

        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.resize(label, self.target_size)

        # Convert to float and normalize
        image = image.astype(np.float32) / 255.0
        label = label.astype(np.float32) / 255.0
        
        # Apply transforms
        if self.image_transform:
            image = self.image_transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1)  # HWC to CHW
            
        if self.label_transform:
            label = self.label_transform(label)
        else:
            label = torch.from_numpy(label).unsqueeze(0)  # Add channel dim
            
        return image, label

# Image and label transforms
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

label_transform = transforms.Compose([
    transforms.ToTensor()
])

# Dataset and Dataloader
train_dataset = EdgeDetectionDataset(
    image_dir='D:/My Project/auto_trace/learning/tes_dataset/train/images',
    label_dir='D:/My Project/auto_trace/learning/tes_dataset/train/labels',
    image_transform=image_transform,
    label_transform=label_transform,
    target_size=(256, 256)
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Sample Check
images, labels = next(iter(train_loader))
print("Image shape:", images.shape)
print("Label shape:", labels.shape)

# Visualization
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.imshow(images[0].permute(1,2,0).cpu().numpy())
plt.title('Image')
plt.axis('off')

plt.subplot(122)
plt.imshow(labels[0][0].cpu().numpy(), cmap='gray')
plt.title('Label')
plt.axis('off')
plt.show()

# ========================
# U-Net Model
# ========================
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
        # Downsampling
        self.down1 = DoubleConv(3, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)

        # Upsampling
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up_conv1 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_conv2 = DoubleConv(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up_conv3 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def crop_and_concat(self, upsampled, bypass):
        _, _, h, w = upsampled.shape
        bypass = transforms.CenterCrop([h, w])(bypass)
        return torch.cat([upsampled, bypass], dim=1)
    
    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x4 = self.down4(self.pool(x3))

        x = self.up1(x4)
        x = self.crop_and_concat(x, x3)
        x = self.up_conv1(x)

        x = self.up2(x)
        x = self.crop_and_concat(x, x2)
        x = self.up_conv2(x)

        x = self.up3(x)
        x = self.crop_and_concat(x, x1)
        x = self.up_conv3(x)

        x = self.final_conv(x)
        return self.sigmoid(x)

# ========================
# Loss Function
# ========================
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

# ========================
# Train Script
# ========================
model = EdgeDetectionModel()
criterion = DiceBCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


val_dataset = EdgeDetectionDataset(
    image_dir='D:/My Project/auto_trace/learning/tes_dataset/val/images',
    label_dir='D:/My Project/auto_trace/learning/tes_dataset/val/labels',
    image_transform=image_transform,
    label_transform=label_transform,
    target_size=(256, 256)
)

val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            if outputs.shape != labels.shape:
                if labels.size(1) == 3:
                    labels = labels.mean(dim=1, keepdim=True)
            
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    avg_loss = val_loss / len(dataloader)
    print(f'Validation Loss: {avg_loss:.4f}')
    return avg_loss



def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=25):
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            if outputs.shape != labels.shape:
                if labels.size(1) == 3:
                    labels = labels.mean(dim=1, keepdim=True)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}] Training Loss: {avg_train_loss:.4f}')
        
        # Validate
        validate(model, val_loader, criterion, device)


# Run training
train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=25)

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
    image = cv2.resize(image, (256, 256))  # Must match training size
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
prediction = predict_edge(model, img_path, image_transform, device)

plt.imshow(prediction, cmap='gray')
plt.title('Predicted Edges')
plt.axis('off')
plt.show()
