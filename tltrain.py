import sys, getopt
from helper import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from torch.optim import Adam
import time
from Bio import SeqIO
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split

try:
    opts, args = getopt.getopt(sys.argv[1:], "m:i:l:n:o:")
except getopt.GetoptError:
    sys.exit(2)

for opt, arg in opts:
    if opt == "-h":
        sys.exit()
    elif opt in ("-m", "--model"):
        modelpath = arg
    elif opt in ("-i", "--input"):
        inputset = arg
    elif opt in ("-l", "--labels"):
        labelset = arg
    elif opt in ("-n", "--newmodel"):
        newModelPath = arg
    elif opt in ("-o", "--out"):
        result_path = arg

INIT_LR = 1e-3
BATCH_SIZE = 1024
EPOCHS = 30

# Load AMAISE onto GPUs
model = TCN()
model = nn.DataParallel(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.read_csv(labelset).to_numpy()
trainData = []


def encodeLabel(num):
    encoded_l = np.zeros(6)
    encoded_l[num] = 1
    # print(num, encoded_l)
    return encoded_l


i = 0
X = []
y = []
for seq in SeqIO.parse(inputset, "fasta"):
    add_len = 9000
    encoded = generate_long_sequences(seq + "0" * add_len)[:add_len]
    label = encodeLabel(train_df[i][1])
    # trainData.append((encoded, label))
    X.append(encoded)
    y.append(label)
    i += 1

# val_size = 10000
# train_size = len(trainData) - val_size
# train_data,val_data = random_split(trainData,[train_size,val_size])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
train_data = []

for i in range(len(X_train)):
    train_data.append((X_train[i], y_train[i]))
val_data = []
for i in range(len(X_val)):
    val_data.append((X_val[i], y_val[i]))

# initialize the train data loader
trainDataLoader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
# initialize the validation data loader
valDataLoader = DataLoader(val_data, shuffle=True, batch_size=BATCH_SIZE)
# calculate steps per epoch for training and validation set
trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
valSteps = len(valDataLoader.dataset) // BATCH_SIZE


model.load_state_dict(torch.load(modelpath, device))
model.module.fc = nn.Linear(model.module.fc.in_features, 6)

for param in model.parameters():
    param.requires_grad = False
for param in model.module.fc.parameters():
    param.requires_grad = True
# model.module.fc.requires_grad = True
model.to(device)
opt = Adam(model.parameters(), lr=INIT_LR)
lossFn = nn.CrossEntropyLoss()
# measure how long training is going to take
print("training the network...")
startTime = time.time()
max_val_acc = 0
file = open(result_path, "a")
# loop over our epochs
for e in range(0, EPOCHS):

    # set the model in training mode
    model.train()

    # Initialize total_loss for each epoch
    total_loss = 0.0
    correct_train_predictions = 0

    # loop over the training set
    for step, (x, y) in enumerate(trainDataLoader):
        # send the input to the device
        (x, y) = (x.clone().detach().float().to(device), y.to(device))
        # perform a forward pass and calculate the training loss
        pred = model(x)
        loss = lossFn(pred, y)

        total_loss += loss.item()
        _, predicted_labels = torch.max(pred, 1)
        _, true_labels = torch.max(y, 1)

        correct_train_predictions += (predicted_labels == true_labels).sum().item()

        # zero out the gradients, perform the backpropagation step,
        # and update the weights
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()

    # Initialize total validation loss for each epoch
    total_val_loss = 0.0
    correct_val_predictions = 0

    # loop over the validation set
    with torch.no_grad():
        for val_x, val_y in valDataLoader:
            val_x, val_y = val_x.clone().detach().float().to(device), val_y.to(device)

            # perform a forward pass and calculate the validation loss
            val_pred = model(val_x)
            val_loss = lossFn(val_pred, val_y)
            total_val_loss += val_loss.item()

            _, predicted_val_labels = torch.max(val_pred, 1)
            _, true_val_labels = torch.max(val_y, 1)
            correct_val_predictions += (
                (predicted_val_labels == true_val_labels).sum().item()
            )

    train_accuracy = correct_train_predictions / len(trainDataLoader.dataset)
    val_accuracy = correct_val_predictions / len(valDataLoader.dataset)
    # calculate the average training and validation loss
    avgTrainLoss = total_loss / trainSteps
    avgValLoss = total_val_loss / valSteps
    print(
        f"Epoch {e+1}/{EPOCHS}, Training Loss: {avgTrainLoss}, Train Accuracy: {train_accuracy}, Validation Loss: {avgValLoss}, Validation Accuracy: {val_accuracy}"
    )
    file.write(
        f"Epoch {e+1}/{EPOCHS}, Training Loss: {avgTrainLoss}, Train Accuracy: {train_accuracy}, Validation Loss: {avgValLoss}, Validation Accuracy: {val_accuracy}\n"
    )
    if max_val_acc < val_accuracy:
        max_val_acc = val_accuracy
        torch.save(model.state_dict(), newModelPath)
# finish measuring how long training took
endTime = time.time()
print(
    "total time taken to train the model: {:.2f} min".format((endTime - startTime) / 60)
)
file.write(
    "total time taken to train the model: {:.2f} min\n".format(
        (endTime - startTime) / 60
    )
)
file.close()
# serialize the model to disk
# modelP = nn.DataParallel(model)
# torch.save(model.state_dict(), newModelPath)
