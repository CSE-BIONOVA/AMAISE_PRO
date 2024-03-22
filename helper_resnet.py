import torch
import random
import torch.nn as nn
from constants import *

class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, filter_size):
    super().__init__()
    self.filter_size = filter_size
    self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=self.filter_size, padding=(self.filter_size - 1)//2, padding_mode='zeros')
    self.bn1 = nn.BatchNorm1d(out_channels)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=self.filter_size, padding=(self.filter_size - 1)//2, padding_mode='zeros')
    self.bn2 = nn.BatchNorm1d(out_channels)

    # Shortcut connection (identity mapping) with padding if necessary
    self.shortcut = nn.Identity()
    if in_channels != out_channels:
      self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)

  def forward(self, x):
    out = self.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    # out += self.shortcut(x)  # Add residual connection
    out = self.relu(out)
    return out

class ResNetTCN(nn.Module):
    def __init__(self):  # Same initialization arguments as TCN
        super().__init__()
        num_input_channels = 4
        num_output_channels = 128
        filter_size = 15  # Define filter size
        pool_amt = 5  # Define pooling amount
        num_classes = 6

        # Replace convolutional layers with residual blocks
        self.block1 = ResidualBlock(num_input_channels, num_output_channels, filter_size)
        self.block2 = ResidualBlock(num_output_channels, num_output_channels, filter_size)
        
        self.fc = nn.Linear(num_output_channels, num_classes)
        self.pool = nn.AvgPool1d(pool_amt)
        self.pad = nn.ConstantPad1d((pool_amt - 1)//2 + 1, 0)
        
        self.filter_size = filter_size
        self.pool_amt = pool_amt
        
    def forward(self, x):
        x = x.transpose(2, 1)
        
        old_shape = x.shape[2]
        if x.shape[2] < self.pool_amt:
            x = self.pad(x)
        new_shape = x.shape[2]
        
        output = self.block1(x)
        output = torch.relu(output)
        output = self.pool(output)*(new_shape/old_shape)
        
        old_shape = output.shape[2]
        if output.shape[2] < self.pool_amt:
            output = self.pad(output)
        new_shape = output.shape[2]
                
        output = self.block2(output)
        output = torch.relu(output)
        # output = self.pool(output)*(new_shape/old_shape)
        
        # old_shape = output.shape[2]
        # if output.shape[2] < self.pool_amt:
        #     output = self.pad(output)
        # new_shape = output.shape[2]
                
        # output = self.c_in3(output)
        # output = torch.relu(output)
        # output = self.pool(output)*(new_shape/old_shape)
        
        # old_shape = output.shape[2]
        # if output.shape[2] < self.pool_amt:
        #     output = self.pad(output)
        # new_shape = output.shape[2]
                
        # output = self.c_in4(output)
        # output = torch.relu(output)
        
        last_layer = nn.AvgPool1d(output.size(2))
        output = last_layer(output).reshape(output.size(0), output.size(1))*(new_shape/old_shape)
        output = self.fc(output)
        return output
   
 