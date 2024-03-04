from helper import *
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import Adam
import time
import sys, getopt
import logging
import click
import matplotlib.pyplot as plt
from Bio.SeqIO.QualityIO import FastqGeneralIterator
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np

@click.command()
@click.option(
    "--input", "-i",
    help = "path to training data",
    type = click.Path(exists = True),
    required = True
)
@click.option(
    "--labels", "-l",
    help = "path to labels of training data",
    type = click.Path(exists = True),
    required = True
)
@click.option(
    "--model", "-m",
    help = "path to save model",
    type = str,
    required = True
)
@click.option(
    "--output", "-o",
    help = "path to save output",
    type = str,
    required = True
)
@click.option(
    "--batch_size", "-b",
    help = "batch size",
    type = int,
    default = 1024,
    show_default = True,
    required = False,
)
@click.option(
    "--epoches", "-e",
    help = "number of epoches",
    type = int,
    default = 50,
    show_default = True,
    required = False,
)
@click.option(
    "--learning_rate", "-lr",
    help = "learning rate",
    type = float,
    default = 0.001,
    show_default = True,
    required = False,
)
@click.help_option('--help', "-h", help = "Show this message and exit")
def main(input, labels, model, output, batch_size, epoches, learning_rate):
    
    newModelPath = model
    inputset = input
    labelset = labels
    resultPath = output
    batchSize = batch_size
    epoches = epoches
    learningRate = learning_rate
    
    logger = logging.getLogger(f"amaisepro")
    logger.setLevel(logging.DEBUG)
    logging.captureWarnings(True)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    consoleHeader = logging.StreamHandler()
    consoleHeader.setFormatter(formatter)
    consoleHeader.setLevel(logging.INFO)
    logger.addHandler(consoleHeader)
    
    fileHandler = logging.FileHandler(f"{resultPath}.log")
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    
    logger.info(f"Model path: {newModelPath}")
    logger.info(f"Input path: {inputset}")
    logger.info(f"Labels path: {labelset}")
    logger.info(f"Results path: {resultPath}")
    
    INIT_LR = learningRate
    logger.info(f"Learning rate: {INIT_LR}")
    BATCH_SIZE = int(batchSize)
    logger.info(f"Batch size: {BATCH_SIZE}")
    EPOCHES = epoches
    logger.info(f"# Epoches: {EPOCHES}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Device: {device}")
    
    train_df = pd.read_csv(labelset).to_numpy()
    
    i = 0
    X = []
    y = []
    
    logger.info("parsing data...")
    
    startTime = time.time()
    
    for seq in SeqIO.parse(inputset, "fasta"):
        add_len = 9000
        lenOfSeq = len(seq)
        if (lenOfSeq-add_len) > 0:
            encoded = generate_long_sequences(seq[:add_len]) 
        else:
            add_len = add_len - lenOfSeq
            encoded = generate_long_sequences(seq + "0"*add_len )
        label = encodeLabel(train_df[i][1])
        X.append(encoded)
        y.append(label)

        i+=1
    
    endTime = time.time()
    logger.info("Total time taken to parse data: {:.2f} min".format({endTime - startTime}/60))
    
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
    file = open(resultPath,'a')
    # initialize the TCN model
    logger.info("initializing the Deep CNN model...")
    model = nn.DataParallel(CNNModel(6)).to(device)

    # initialize our optimizer and loss function
    opt = Adam(model.parameters(), lr=INIT_LR)
    lossFn = nn.CrossEntropyLoss()
    # measure how long training is going to take
    logger.info("training the network...")
    startTime = time.time()
    max_val_acc = 0
    
    train_losses = []
    train_accuracies = []
    validation_losses = []
    validation_accuracies = []
    epoch_list = [e+1 for e in range(0, EPOCHES)]
    
    # loop over our epoches
    for e in range(0, EPOCHES):
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
            # pred = torch.softmax(model(x). dim=1)
            pred = model(x)
            loss = lossFn(pred, y)
            total_loss += loss.item() # Accumulate the loss for the epoch
            # zero out the gradients, perform the backpropagation step,
            # and update the weights
            # calculate correct predictions
            _, predicted_lables = torch.max(pred, 1)
            _, true_labels = torch.max(y, 1)
            correct_train_predictions += (predicted_lables ==true_labels).sum().item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            # opt.zero_grad()
            
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
                correct_val_predictions += (predicted_val_labels == true_val_labels).sum().item()
                
        # Calculate and print the validation loss for the epoch
        # avg_val_loss = total_val_loss / len(valDataLoader)
        train_accuracy = correct_train_predictions / len(trainDataLoader.dataset)
        val_accuracy = correct_val_predictions / len(valDataLoader.dataset)
        total_val_loss = total_val_loss / len(valDataLoader.dataset)
        total_loss = total_loss / len(trainDataLoader.dataset)
        logger.info(f'Epoch {e+1}/{EPOCHES}, Training Loss: {total_loss}, Train Accuracy: {train_accuracy}, Validation Loss: {total_val_loss}, Validation Accuracy: {val_accuracy}')
        
        # Save accuracies and losses
        train_losses.append(total_loss)
        train_accuracies.append(train_accuracy)
        validation_losses.append(total_val_loss)
        validation_accuracies.append(val_accuracy)
        
        if max_val_acc < val_accuracy:
            max_val_acc = val_accuracy
            torch.save(model.state_dict(), newModelPath)
    # finish measuring how long training took
    endTime = time.time()
    logger.info("total time taken to train the model: {:.2f} min".format((endTime - startTime)/60))
    
    # serialize the model to disk
    # modelP = nn.DataParallel(model)
    # torch.save(modelP.state_dict(), newModelPath)
    
    # Plot training validation losses vs. epoches
    
    plt.plot(epoch_list, train_losses, label="Training loss")
    plt.plot(epoch_list, validation_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
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
    
def encodeLabel(num):
    encoded_l = np.zeros(6)
    encoded_l[num] = 1
    return encoded_l

if __name__ == "__main__":
    main()    
        