import torch
import torchvision.transforms as transforms
from ContrastiveLoss import ContrastiveLoss
from dataset.signature.SignatureDataset import SignatureDataset
from SiameseNetwork import SiameseNetwork


siamese_dataset = SignatureDataset(
    transform=transforms.Compose(
        [
            transforms.Resize((18,128)),
            transforms.ToTensor()
        ]
    )
)
print(len(siamese_dataset))

#set the device to cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

siamese_netword = SiameseNetwork(device=device)
loss = siamese_netword.train(dataset=siamese_dataset)
print(loss)
