import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import TransformerNet
from losses import StyleLoss, ContentLoss, TotalVariationLoss
from PIL import Image

# Configurações
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
epochs = 10
style_weight = 1e5
content_weight = 1e0
tv_weight = 1e-6

# Carregar imagem de estilo
style_image = Image.open("data/higurashiNoNakuNi.jpg").convert("RGB")
style_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
style_tensor = style_transform(style_image).unsqueeze(0).to(device)

# Dataset de conteúdo
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = datasets.ImageFolder("../../../train2017/coco20/", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Modelo e otimizador
model = TransformerNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Funções de perda
style_loss_fn = StyleLoss().to(device)
content_loss_fn = ContentLoss().to(device)
tv_loss_fn = TotalVariationLoss().to(device)

# Treinamento
for epoch in range(epochs):
    for batch_idx, (content_batch, _) in enumerate(dataloader):
        content_batch = content_batch.to(device)
        
        # Gerar imagem estilizada
        stylized_batch = model(content_batch)
        
        # Calcular perdas
        style_loss = style_loss_fn(stylized_batch, style_tensor) * style_weight
        content_loss = content_loss_fn(stylized_batch, content_batch) * content_weight
        tv_loss = tv_loss_fn(stylized_batch) * tv_weight
        
        total_loss = style_loss + content_loss + tv_loss
        
        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if batch_idx % 50 == 0:
            print(f"Epoch: {epoch+1}/{epochs} | Batch: {batch_idx} | Loss: {total_loss.item()}")

# Salvar modelo
torch.save(model.state_dict(), "models/anime_style.pth")