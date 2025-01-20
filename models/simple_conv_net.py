import torch
import torch.nn as nn



class SimpleConvNet(nn.Module):
    def __init__(self, input_dim=(3,28,28), 
                 conv_params={"kernel_num":10, "kernel_size":3, "padding":0, "stride":1}, 
                 hidden_size=10, output_size=10):
        super().__init__()

        C, H, W = input_dim
        kernel_num, kernel_size, padding, stride = \
        conv_params["kernel_num"], conv_params["kernel_size"], conv_params["padding"], conv_params["stride"]
        conv_output_size = (H + 2 * padding - kernel_size) // stride + 1
        pool_output_size = (conv_output_size // 2) * (conv_output_size // 2) * kernel_num        

        self.conv = nn.Conv2d(in_channels=C, out_channels=kernel_num, \
                              kernel_size=kernel_size, stride=stride, padding=padding )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(pool_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        

    def forward(self, x):

        x = self.conv(x)
        x = self.pool(x)
        x = self.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
    

