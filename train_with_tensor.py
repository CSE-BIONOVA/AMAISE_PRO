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
import psutil

@click.command()
@click.option(
    "--input", "-i",
    help = "path to training data tensors",
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
@click.option(
    "--max_length", "-ml",
    help = "maximum length of sequence",
    type = int,
    default = 9000,
    show_default = True,
    required = False,
)
@click.help_option('--help', "-h", help = "Show this message and exit")

def main(input, model, output, batch_size, epoches, learning_rate, max_length):
    
    newModelPath = model
    inputset = input
    resultPath = output
    batchSize = batch_size
    epoches = epoches
    learningRate = learning_rate
    max_length = max_length
    
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
    logger.info(f"Results path: {resultPath}")
    
    INIT_LR = learningRate
    BATCH_SIZE = int(batchSize)
    EPOCHES = epoches
    
    logger.info(f"Learning rate: {INIT_LR}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"# Epoches: {EPOCHES}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Device: {device}")
    

    logger.info("parsing data...")
    input_tensor_dict = torch.load(inputset)
    print(len(input_tensor_dict['X']))
    print(len(input_tensor_dict['y']))
    X = input_tensor_dict['X']
    y = input_tensor_dict['y']
    # startTime = time.time()
    
    # for seq in SeqIO.parse(inputset, "fasta"):
    #     add_len = max_length
    #     lenOfSeq = len(seq)
    #     if (lenOfSeq-add_len) > 0:
    #         encoded = generate_onehot_encoding(seq[:add_len]) 
    #     else:
    #         add_len = add_len - lenOfSeq
    #         encoded = generate_onehot_encoding(seq + "N"*add_len )
    #     label = encodeLabel(train_label_dict[seq.id])
    #     X.append(encoded)
    #     y.append(label)
    
    
    # endTime = time.time()
    # encoding_time_diff = (endTime - startTime)/60
    # logger.info(f"Total time taken to parse data: {encoding_time_diff} min")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

    train_data = []
    for i in range(len(X_train)):
        train_data.append((X_train[i], y_train[i]))
    val_data = []
    for i in range(len(X_val)):
        val_data.append((X_val[i], y_val[i]))
        
   
    trainDataLoader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    valDataLoader = DataLoader(val_data, shuffle=True, batch_size=BATCH_SIZE)

    trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
    valSteps = len(valDataLoader.dataset) // BATCH_SIZE
    
    logger.info("initializing the Deep CNN model...")
    model = nn.DataParallel(DeepCNN(max_len=max_length)).to(device)

    opt = Adam(model.parameters(), lr=INIT_LR)
    lossFn = nn.CrossEntropyLoss()
    
    logger.info("training the network...")
    
    startTime = time.time()
    max_val_acc = 0
    train_losses, train_accuracies = [], []
    validation_losses, validation_accuracies = [], []
    epoch_list = [e+1 for e in range(0, EPOCHES)]
    
    
    for e in range(0, EPOCHES):

        model.train()
    
        total_loss = 0.0
        correct_train_predictions = 0
        
        for step, (x, y) in enumerate(trainDataLoader):
            (x, y) = (x.to(device), y.to(device))
            pred = model(x)
            loss = lossFn(pred, y)
            total_loss += loss.item() # total loss for each epoch
            # zero out the gradients, 
            # perform the backpropagation step,
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
        
        with torch.no_grad():
            for step, (val_x, val_y) in enumerate(valDataLoader):
                val_x, val_y = val_x.to(device), val_y.to(device)
                
                # perform a forward pass and calculate the validation loss
                val_pred = model(val_x)
                val_loss = lossFn(val_pred, val_y)
                total_val_loss += val_loss.item()
                
                _, predicted_val_labels = torch.max(val_pred, 1)
                _, true_val_labels = torch.max(val_y, 1)
                correct_val_predictions += (predicted_val_labels == true_val_labels).sum().item()
                
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
    
    endTime = time.time()
    peak_memory = psutil.Process().memory_info().peak_wset

    logger.info("total time taken to train the model: {:.2f} min".format((endTime - startTime)/60))
    logger.info(f"Peak memory usage: {peak_memory}")

    
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
    
if __name__ == "__main__":
    main()    
        