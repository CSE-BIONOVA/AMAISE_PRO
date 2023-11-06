from helper import *
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import Adam
import time
import sys, getopt
from Bio.SeqIO.QualityIO import FastqGeneralIterator


# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", type=str, required=True, help="path to output trained model")
# ap.add_argument("-i", "--input", type=str, required=True, help="path to train dataset")

# args = vars(ap.parse_args())

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
 
# define training hyperparameters
INIT_LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 30

# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.read_csv(labelset).to_numpy()
trainData = []
i = 0
for seq in FastqGeneralIterator(open(inputset)):
    trainData.append((generate_long_sequences(seq[1]), train_df[i][3]))
    i = i + 1

trainData = trainData[481701:491702]
# for row in train_df:
#     trainData.append((generate_long_sequences(row[0]),row[1]))

# initialize the train data loader
trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=BATCH_SIZE)
# calculate steps per epoch for training set
trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE

# initialize the TCN model
print("initializing the TCN model...")
model = TCN().to(device)
# initialize our optimizer and loss function
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
		(x, y) = (torch.tensor(x).float().to(device), y.to(device))
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
modelP = nn.DataParallel(model)
torch.save(modelP.state_dict(), newModelPath)