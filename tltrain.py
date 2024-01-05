import sys, getopt
from helper import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from torch.optim import Adam
import time
from Bio import SeqIO


try:
    opts, args = getopt.getopt(sys.argv[1:], 'm:i:l:n:')
except getopt.GetoptError:
    sys.exit(2)
    
for opt, arg in opts:
    if opt == '-h':
        sys.exit()
    elif opt in ("-m", "--model"):
        modelpath = arg
    elif opt in ("-i", "--input"):
        inputset = arg
    elif opt in ("-l", "--labels"):
        labelset = arg
    elif opt in ("-n", "--newmodel"):
        newModelPath = arg
 
INIT_LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 30 
        
# Load AMAISE onto GPUs
model = TCN()
model = nn.DataParallel(model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.load_state_dict(torch.load(modelpath, device))
model.module.fc = nn.Linear(model.module.fc.in_features, 2)

for param in model.parameters():
    param.requires_grad = False
for param in model.module.fc.parameters():
    param.requires_grad = True   
# model.module.fc.requires_grad = True
########## data loading

train_df = pd.read_csv(labelset).to_numpy()
trainData = []
i = 0
for seq in SeqIO.parse(inputset, "fasta"):
    if train_df[i][1]!=1:
       trainData.append((generate_long_sequences(seq[1]), 0))
       print("--0--")
    else:
       trainData.append((generate_long_sequences(seq[1]), 1))
       print("--1--")
       i = i + 1
print(i)
trainData = trainData[:20000]
# for row in train_df:
#     trainData.append((generate_long_sequences(row[0]),row[1]))

# initialize the train data loader
trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=BATCH_SIZE)
# calculate steps per epoch for training set
trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE

opt = Adam(model.parameters(), lr=INIT_LR)
lossFn = nn.CrossEntropyLoss()
# measure how long training is going to take
print("training the network...")
startTime = time.time()

# loop over our epochs
for e in range(0, EPOCHS):
	# set the model in training mode
	model.train()
	
	# loop over the training set
	for (x, y) in trainDataLoader:
		# send the input to the device
		(x, y) = (x.clone().detach().float().to(device), y.to(device))
		# perform a forward pass and calculate the training loss
		pred = torch.sigmoid(model(x))
		loss = lossFn(pred, y)

        # zero out the gradients, perform the backpropagation step,
		# and update the weights
		opt.zero_grad()
		loss.backward()
		opt.step()
# finish measuring how long training took
endTime = time.time()
print("total time taken to train the model: {:.2f}s".format(endTime - startTime))

# serialize the model to disk
#modelP = nn.DataParallel(model)
torch.save(model.state_dict(), newModelPath)
