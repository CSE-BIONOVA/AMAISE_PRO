from helper import *
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import Adam
import time
import sys, getopt
from Bio.SeqIO.QualityIO import FastqGeneralIterator
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
from collections import Counter

# import numpy as np

# from torch.utils.tensorboard import SummaryWriter

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", type=str, required=True, help="path to output trained model")
# ap.add_argument("-i", "--input", type=str, required=True, help="path to train dataset")

# args = vars(ap.parse_args())

try:
    opts, args = getopt.getopt(sys.argv[1:], "m:i:l:o:")
except getopt.GetoptError:
    sys.exit(2)

for opt, arg in opts:
    if opt == "-h":
        sys.exit()
    elif opt in ("-m", "--model"):
        newModelPath = arg
    elif opt in ("-i", "--input"):
        inputset = arg
    elif opt in ("-l", "--labels"):
        labelset = arg
    elif opt in ("-o", "--out"):
        result_path = arg

# define training hyperparameters
INIT_LR = 1e-3
BATCH_SIZE = 1024
EPOCHS = 50

# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.read_csv(labelset).to_numpy()


def encodeLabel(num):
    encoded_l = np.zeros(6)
    encoded_l[num] = 1
    return encoded_l


i = 0
X = []
y = []

for seq in SeqIO.parse(inputset, "fasta"):
    add_len = 9000
    lenOfSeq = len(seq)
    if (lenOfSeq - add_len) > 0:
        encoded = generate_long_sequences(seq[:add_len])
    else:
        add_len = add_len - lenOfSeq
        encoded = generate_long_sequences(seq + "0" * add_len)
    label = encodeLabel(train_df[i][1])
    X.append(encoded)
    y.append(label)

    i += 1

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
train_data = []

for i in range(len(X_train)):
    train_data.append((X_train[i], y_train[i]))
val_data = []
for i in range(len(X_val)):
    val_data.append((X_val[i], y_val[i]))

# counts = {}
# for item in y:
#   if str(item) not in counts:
#     counts[str(item)] = 0
#   counts[str(item)] += 1

# print(counts)
# counts = {}
# for item in y_train:
#   if str(item) not in counts:
#     counts[str(item)] = 0
#   counts[str(item)] += 1

# print(counts)
# counts = {}
# for item in y_val:
#   if str(item) not in counts:
#     counts[str(item)] = 0
#   counts[str(item)] += 1

# print(counts)

# initialize the train data loader
trainDataLoader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
# initialize the validation data loader
valDataLoader = DataLoader(val_data, shuffle=True, batch_size=BATCH_SIZE)
# calculate steps per epoch for training and validation set
trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
valSteps = len(valDataLoader.dataset) // BATCH_SIZE
file = open(result_path, "a")
# initialize the TCN model
print("initializing the TCN model...")
model = nn.DataParallel(TCN()).to(device)

# initialize our optimizer and loss function
opt = Adam(model.parameters(), lr=INIT_LR)
lossFn = nn.CrossEntropyLoss()
# measure how long training is going to take
print("training the network...")
startTime = time.time()
max_val_acc = 0
train_losses, train_accuracies = [], []
validation_losses, validation_accuracies = [], []
epoch_list = [e+1 for e in range(0, EPOCHES)]

# loop over our epochs
for e in range(0, EPOCHS):
    # set the model in training mode
    model.train()

    # Initialize total_loss for each epoch
    total_loss = 0.0
    correct_train_predictions = 0

    # loop over the training set
    for step, (x, y) in enumerate(trainDataLoader):
        # for (x, y) in trainDataLoader:
        # send the input to the device
        (x, y) = (x.clone().detach().float().to(device), y.to(device))
        # perform a forward pass and calculate the training loss
        # pred = torch.softmax(model(x),dim=1)
        pred = model(x)
        loss = lossFn(pred, y)
        # print(loss.item())  # Print the current loss value
        total_loss += loss.item()  # Accumulate the loss for the epoch
        # zero out the gradients, perform the backpropagation step,
        # and update the weights
        # calculate correct predictions
        _, predicted_labels = torch.max(pred, 1)
        _, true_labels = torch.max(y, 1)
        correct_train_predictions += (predicted_labels == true_labels).sum().item()
        opt.zero_grad()
        loss.backward()
        opt.step()
        # opt.zero_grad()
    # print(f'Epoch {e+1}/{EPOCHS}, Total Training Loss: {total_loss}')

    model.eval()

    # Initialize total validation loss for each epoch
    total_val_loss = 0.0
    correct_val_predictions = 0

    # loop over the validation set
    with torch.no_grad():
        for step, (val_x, val_y) in enumerate(valDataLoader):
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

    # Calculate and print the validation loss for the epoch
    # avg_val_loss = total_val_loss / len(valDataLoader)
    train_accuracy = correct_train_predictions / len(trainDataLoader.dataset)
    val_accuracy = correct_val_predictions / len(valDataLoader.dataset)
    total_val_loss = total_val_loss / len(valDataLoader.dataset)
    total_loss = total_loss / len(trainDataLoader.dataset)
    print(
        f"Epoch {e+1}/{EPOCHS}, Training Loss: {avgTrainLoss}, Train Accuracy: {train_accuracy}, Validation Loss: {avgValLoss}, Validation Accuracy: {val_accuracy}"
    )
    file.write(
        f"Epoch {e+1}/{EPOCHS}, Training Loss: {avgTrainLoss}, Train Accuracy: {train_accuracy}, Validation Loss: {avgValLoss}, Validation Accuracy: {val_accuracy}\n"
    )

    # Save accuracies and losses
    train_losses.append(total_loss)
    train_accuracies.append(train_accuracy)
    validation_losses.append(total_val_loss)
    validation_accuracies.append(val_accuracy)

    if max_val_acc < val_accuracy:
        max_val_acc = val_accuracy
        # modelP = nn.DataParallel(model)
        torch.save(model.state_dict(), newModelPath)
# finish measuring how long training took
endTime = time.time()
peak_memory = psutil.Process().memory_info().peak_wset

print(
    "total time taken to train the model: {:.2f} min".format((endTime - startTime) / 60)
)
print(f"Peak memory usage: {peak_memory}")
file.write(
    "total time taken to train the model: {:.2f} min\n".format(
        (endTime - startTime) / 60
    )
)
file.close()

# Plot training validation losses vs. epoches
    
plt.plot(epoch_list, train_losses, label="Training loss")
plt.plot(epoch_list, validation_losses, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="upper right")
plt.savefig(f"{resultPath}_losses.png", dpi=300, bbox_inches='tight')

plt.clf()

# Plot training validation accuracies vs. epoches

plt.plot(epoch_list, train_accuracies, label="Training accuracy")
plt.plot(epoch_list, validation_accuracies, label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="lower right")
plt.savefig(f"{resultPath}_accuracies.png", dpi=300, bbox_inches='tight')
# # serialize the model to disk
# modelP = nn.DataParallel(model)
# torch.save(modelP.state_dict(), newModelPath)