import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ContrastiveLoss import ContrastiveLoss

class SiameseNetwork(nn.Module):
    def __init__(self, device) -> None:
        super(SiameseNetwork, self).__init__()
        self.device = device
        self.to(self.device)
        self.epochs = -1
        self.loss_values = []
        
        # Defining the CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=12,stride=1),              #(116, 116, 96)
            nn.ReLU(inplace=True),                                  #(116, 116, 96)
            nn.MaxPool2d(kernel_size=3, stride=2),                  #(58, 58, 96)
            
            nn.Conv2d(96, 256, kernel_size=6,stride=1,padding=2),   #(56, 56, 256)
            nn.ReLU(inplace=True),                                  #(56, 56, 256)
            nn.MaxPool2d(3, stride=2),                              #(28, 28, 256)

            nn.Conv2d(256,384 , kernel_size=3,stride=1,padding=1),  #(27, 27, 384)
            nn.ReLU(inplace=True),                                  #(27, 27, 384)
            
            nn.Conv2d(384,256 , kernel_size=3,stride=1,padding=1),  #(26, 26, 256)
            nn.ReLU(inplace=True),                                  #(26, 26, 256)
            nn.MaxPool2d(3, stride=2),                              #(13, 13, 256)
        )
        
        # Defining the fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(43264, 1024),
            nn.ReLU(inplace=True),
            
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128, 16)
        )

    def forward_once(self, x):
        # Forward pass 
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        # Initialize output tensors
        output1 = torch.empty(input1.size(0), dtype=input1.dtype, device=input1.device)
        output2 = torch.empty(input2.size(0), dtype=input2.dtype, device=input2.device)

        # Forward pass of input1 on the default GPU stream
        with torch.cuda.stream(torch.cuda.default_stream()):
            output1 = self.forward_once(input1)

        # Forward pass of input2 on the default GPU stream
        with torch.cuda.stream(torch.cuda.default_stream()):
            output2 = self.forward_once(input2)

        # Synchronize the default stream to ensure outputs are ready
        torch.cuda.synchronize()

        return output1, output2
    
    def train(self, dataloader: DataLoader=None, loss_function:torch.nn.Module=ContrastiveLoss):
        torch.cuda.empty_cache()
        if (dataloader is None):
            return
        
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=0.0005)
        
        for epoch in range(self.epochs + 1, self.epochs + 10):
            self.epochs = epoch
            for _, data in enumerate(dataloader, 0):
                data.to(self.device)
                img0, img1 , label = data
                img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
                
                optimizer.zero_grad()
                output1,output2 = self(img0, img1)
                
                loss_contrastive = loss_function(output1,output2,label)
                # compute gradient 
                loss_contrastive.backward()
                self.loss_values.append(loss_contrastive.item())
                optimizer.step()
            
            self.save()
            print("Epoch {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
            
        return self.loss_values

    def save(self):
        torch.save(self.state_dict(), "saved_model/Siamese_epochs_{}.pt".format(self.epochs))
        print("Model Saved Successfully") 
        
    def load(self, path):
        self = torch.load(path)
        