from helper import *
import torch
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn
import time
import logging
import click
import numpy as np
import psutil
from sklearn.metrics import classification_report


@click.command()
@click.option(
    "--input",
    "-i",
    help="path to testing data",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--labels",
    "-l",
    help="path to labels of testing data",
    type=click.Path(exists=True),
    required=True,
)
@click.option("--model", "-m", help="path to the existing model", type=str, required=True)
@click.option("--output", "-o", help="path to save output", type=str, required=True)
@click.help_option("--help", "-h", help="Show this message and exit")
def main(input, labels, model, output):

    modelPath = model
    inputset = input
    labelset = labels
    resultPath = output

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

    logger.info(f"Model path: {modelPath}")
    logger.info(f"Input path: {inputset}")
    logger.info(f"Labels path: {labelset}")
    logger.info(f"Results path: {resultPath}")

    BATCH_SIZE = 2048

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Device: {device}")

    test_df = pd.read_csv(labelset)
    test_data_arr = pd.read_csv(inputset, header=None).to_numpy()

    test_data = []

    logger.info("parsing data...")

    startTime = time.time()

    i = 0
    for row in test_data_arr:
        test_data.append(
            (
                np.reshape(row.astype(np.float32), (-1, 1)),
                encodeLabel(test_df["y_true"][i]),
            )
        )
        i = i + 1

    endTime = time.time()
    encoding_time_diff = (endTime - startTime) / 60
    logger.info(f"Total time taken to parse data: {encoding_time_diff} min")

    testDataLoader = DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE)

    logger.info("initializing and loading the TCN model...")
    model = nn.DataParallel(TCN())
    model.load_state_dict(torch.load(modelPath, device))
    model = model.to(device)
    model.eval()

    logger.info("predicting the classes...")

    startTime = time.time()
    
    correct_test_predictions = 0
    predicted = []
    true = []

    with torch.no_grad():
        for step, (test_x, test_y) in enumerate(testDataLoader):
            test_x, test_y = test_x.to(device), test_y.to(device)
            pred = torch.nn.functional.softmax(model(test_x), dim=1)
            _, predicted_labels = torch.max(pred, 1)
            _, true_labels = torch.max(test_y, 1)
            predicted.extend(predicted_labels.cpu())
            true.extend(true_labels.cpu())
            correct_test_predictions += (predicted_labels == true_labels).sum().item()

    test_accuracy = correct_test_predictions / len(testDataLoader.dataset)
    logger.info(f"Test Accuracy: {test_accuracy}")

    logger.info(f'\n {classification_report(true,predicted,target_names=["Host", "Bacteria", "Virus", "Fungi", "Archaea", "Protozoa"],)}')
    endTime = time.time()
    memory = psutil.Process().memory_info()

    logger.info(
        "total time taken to predict results: {:.2f} min".format(
            (endTime - startTime) / 60
        )
    )
    logger.info(f"Memory usage: {memory}")

if __name__ == "__main__":
    main()
