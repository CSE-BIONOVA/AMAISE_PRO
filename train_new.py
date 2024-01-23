from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from helper import *
import argparse
import pandas as pd
from torch.optim import Adam
import time
import sys, getopt
from Bio.SeqIO.QualityIO import FastqGeneralIterator

try:
    opts, args = getopt.getopt(sys.argv[1:], 'm:i:l:')
except getopt.GetoptError:
    sys.exit(2)
    
for opt, arg in opts:
    if opt == '-h':
        sys.exit()
    elif opt in ("-m", "--model"):
        newModelPath = arg
    elif opt in ("-i", "--input"):
        inputset = arg
    elif opt in ("-l", "--labels"):
        labelset = arg
 
# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.read_csv(labelset).to_numpy()
trainData = []
i = 0
x = 0
y = 0
for seq in SeqIO.parse(inputset, "fasta"):
    if train_df[i][1]!=1:
          trainData.append((generate_long_sequences(seq[1]), 0))
          x +=1
          
    else:
          trainData.append((generate_long_sequences(seq[1]), 1))
          y+=1
    i+=1

print(x)
print(y)

trainData = trainData[:30000]
a = 0
b = 0
for x in trainData:
	if x[1]==0:
		a = a + 1
	else:
		b = b + 1
print(a)
print(b)

batch_size = 64
val_size = 5000
train_size = len(trainData) - val_size 

train_data,val_data = random_split(trainData,[train_size,val_size])
print(f"Length of Train Data : {len(train_data)}")
print(f"Length of Validation Data : {len(val_data)}")

#load the train and validation into batches.
train_dl = DataLoader(train_data, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
val_dl = DataLoader(val_data, batch_size*2, num_workers = 4, pin_memory = True)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

  
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

  
def fit(epochs, lr, model, train_loader, val_loader, opt_func = torch.optim.SGD):
    
    history = []
    optimizer = opt_func(model.parameters(),lr)
    for epoch in range(epochs):
        
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    
    return history

num_epochs = 30
BATCH_SIZE = 64

opt_func = torch.optim.Adam
lr = 1e-3#fitting the model on training data and record the result after each epoch

print("initializing the TCN model...")
model = TCN().to(device)
history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)