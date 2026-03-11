# -*- coding: utf-8 -*-
"""
make_model v02.py

Deskripsi:
Program ini melatih model deep learning untuk deteksi tepi (edge detection)
menggunakan arsitektur mirip U-Net. Program ini menggunakan PyTorch dan
secara otomatis mendeteksi serta memanfaatkan GPU (CUDA) jika tersedia,
sehingga proses pelatihan dapat berjalan lebih cepat.

Fungsi utama:
1. Mengelola dataset gambar dan label untuk pelatihan dan validasi.
2. Mendefinisikan arsitektur model 'EdgeDetectionModel' yang merupakan
   varian dari U-Net.
3. Menggunakan fungsi loss 'DiceBCELoss' yang menggabungkan Binary Cross-Entropy
   dan Dice Loss, cocok untuk segmentasi gambar.
4. Melatih model, memantau loss pada set validasi, dan menyimpan model terbaik.

Cara menjalankan:
1. Pastikan PyTorch dengan dukungan CUDA sudah terinstal.
2. Siapkan dataset gambar dan label di direktori yang sesuai (sesuaikan path).
3. Jalankan skrip ini menggunakan interpreter Python.
4. Program akan secara otomatis mendeteksi GPU. Jika ada, pelatihan akan
   berjalan di GPU.

Dependencies:
- torch
- torchvision
- opencv-python
- numpy
- matplotlib
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

# --- Preprocessing Functions ---

def pad_to_multiple(img, multiple=16):
    """
    Menambahkan padding pada gambar agar tinggi dan lebarnya menjadi kelipatan dari 'multiple'.
    Ini diperlukan agar dimensi gambar kompatibel dengan arsitektur U-Net.
    
    Args:
        img (numpy.ndarray): Gambar input.
        multiple (int): Kelipatan yang diinginkan.
    
    Returns:
        numpy.ndarray: Gambar yang sudah diberi padding.
    """
    h, w = img.shape[:2]
    new_h = (h + multiple - 1) // multiple * multiple
    new_w = (w + multiple - 1) // multiple * multiple
    pad_h = new_h - h
    pad_w = new_w - w
    return cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

# --- Dataset and DataLoader ---

class EdgeDetectionDataset(Dataset):
    """
    Custom Dataset untuk memuat gambar dan label untuk deteksi tepi.
    """
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
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

        # Apply padding
        image = pad_to_multiple(image)
        label = pad_to_multiple(label)

        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        label = label.astype(np.float32) / 255.0

        # Apply transforms and convert to PyTorch tensors
        if self.transform:
            image = self.transform(image)
            label = self.transform(label).unsqueeze(0) # unsqueeze for single channel
        else:
            image = torch.from_numpy(image).permute(2, 0, 1)
            label = torch.from_numpy(label).unsqueeze(0)

        return image, label

# --- Model Architecture ---

class DoubleConv(nn.Module):
    """
    Blok konvolusi ganda yang terdiri dari dua lapis Conv2d, BatchNorm2d, dan ReLU.
    """
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
    """
    Arsitektur model U-Net untuk deteksi tepi.
    """
    def __init__(self):
        super().__init__()
        # Downward path
        self.down1 = DoubleConv(3, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.maxpool = nn.MaxPool2d(2)

        # Upward path
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up_conv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up_conv3 = DoubleConv(128, 64)
        
        # Output layer
        self.final_conv = nn.Conv2d(64, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Downward path with skip connections
        x1 = self.down1(x)
        x2 = self.down2(self.maxpool(x1))
        x3 = self.down3(self.maxpool(x2))
        x4 = self.down4(self.maxpool(x3))

        # Upward path with concatenation
        x = self.up1(x4)
        x = self.up_conv1(torch.cat([x, x3], dim=1))
        x = self.up2(x)
        x = self.up_conv2(torch.cat([x, x2], dim=1))
        x = self.up3(x)
        x = self.up_conv3(torch.cat([x, x1], dim=1))
        
        return self.sigmoid(self.final_conv(x))

# --- Loss Function ---

class DiceBCELoss(nn.Module):
    """
    Fungsi loss gabungan antara Binary Cross-Entropy dan Dice Loss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        # Flatten tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Binary Cross-Entropy loss
        bce = F.binary_cross_entropy(inputs, targets)
        
        # Dice Loss
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + 1.) / (inputs.sum() + targets.sum() + 1.)
        
        # Total loss is a combination of BCE and (1 - Dice)
        return bce + (1 - dice)

# --- Training and Validation Loop ---

def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=25):
    """
    Fungsi utama untuk melatih dan mengevaluasi model.
    """
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("Model terbaik disimpan!")

# --- Main Execution Block ---

if __name__ == "__main__":
    # Tentukan device yang akan digunakan (GPU atau CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Menggunakan device: {device}")

    # Konfigurasi path dataset
    train_image_dir = 'C:/Users/RNU/Documents/Kerja Praktek/DATA/data_train/image_train'
    train_label_dir = 'C:/Users/RNU/Documents/Kerja Praktek/DATA/data_train/label_train'
    val_image_dir = 'C:/Users/RNU/Documents/Kerja Praktek/DATA/data_testing/image_test'
    val_label_dir = 'C:/Users/RNU/Documents/Kerja Praktek/DATA/data_testing/label_test'

    # Buat dataset dan dataloader
    transform = transforms.ToTensor()
    train_dataset = EdgeDetectionDataset(train_image_dir, train_label_dir, transform)
    val_dataset = EdgeDetectionDataset(val_image_dir, val_label_dir, transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Optional: Visualisasi sampel data
    print("\nMenampilkan contoh data dari dataset...")
    images, labels = next(iter(train_loader))
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(images[0].permute(1, 2, 0).numpy())
    plt.title('Image')
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(labels[0][0].numpy(), cmap='gray')
    plt.title('Label')
    plt.axis('off')
    plt.show()

    # Inisialisasi model, loss function, dan optimizer
    model = EdgeDetectionModel().to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Mulai proses pelatihan
    print("\nMemulai pelatihan model...")
    train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=25)