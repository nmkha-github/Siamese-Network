import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ContrastiveLoss import ContrastiveLoss

class SiameseNetwork(nn.Module):
    def __init__(self) -> None:
        super(SiameseNetwork, self).__init__()
        self.epochs = -1
        self.loss_values = []
        self.floaten_size = 0
        
        # Defining the CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=12, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=6, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
        ).cuda()
        
        # Defining the fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(0, 1024),
            nn.ReLU(inplace=True),
            
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128, 16)
        ).cuda()

    def forward_once(self, x):
        # Forward pass 
        output = self.cnn(x)
        floaten_tensor = output.view(output.size(0), -1)
        if self.floaten_size == 0:
            self.floaten_size = floaten_tensor.size(1)
            self.fc[0] = nn.Linear(self.floaten_size, 1024)
            self.fc.cuda()
        output = self.fc(floaten_tensor)
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
    
    def train(self, dataloader: DataLoader=None, loss_function:torch.nn.Module=ContrastiveLoss, save_path=''):
        torch.cuda.empty_cache()
        if (dataloader is None):
            return

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=0.0005)
        loss_f = loss_function()
        
        for epoch in range(self.epochs + 1, self.epochs + 10):
            self.epochs = epoch
            
            for _, batch in enumerate(dataloader):
                img0, img1 , label = batch
                img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
                
                optimizer.zero_grad()
                output1,output2 = self(img0, img1)
                
                loss_value = loss_f(output1, output2, label)
                # compute gradient 
                loss_value.backward()
                self.loss_values.append(loss_value.item())
                optimizer.step()
            
            self.save(path=save_path)
            print("Epoch {}\n Current loss {}\n".format(epoch, loss_value.item()))
            
        return self.loss_values

    def save(self, path=''):
        if path != '':
            torch.save(self.state_dict(), path + "/Siamese_epoch_{}.pt".format(self.epochs))
        else:
            torch.save(self.state_dict(), "saved_models/Siamese_epoch_{}.pt".format(self.epochs))
        print("Model Saved Successfully") 
        
    def load(self, path):
        # Load the model's state_dict from the specified path
        state_dict = torch.load(path)

        self.fc[0] = state_dict.__class__.fc[0]
            
        # Load the state_dict onto the device it was saved from
        self.load_state_dict(state_dict)

        # Move the model to the appropriate device (cuda or cpu)
        self.cuda()
        
        print('current epoch: ' + str(self.epochs))

        