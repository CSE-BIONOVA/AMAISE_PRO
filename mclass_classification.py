from helper import *
import torch
import pandas as pd
from Bio import SeqIO
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
    "--input_fasta_fastq",
    "-i",
    help="path to input (fasta/fastq) data",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "type_",
    "-t",
    help="type of the input file (fasta or fastq)",
    type=click.Choice(["fasta", "fastq"]),
    required=True,
)
@click.option(
    "--input_kmers",
    "-k",
    help="path to generated k-mers",
    type=click.Path(exists=True),
    required=True,
)
@click.option("--model", "-m", help="path to the model", type=click.Path(exists=True), default="models/AMAISE_PRO", required=False)
@click.option(
    "--output",
    "-o",
    help="path to a folder to save predictions",
    type=str,
    required=True,
)
@click.help_option("--help", "-h", help="Show this message and exit")
def main(input_fasta_fastq, type_, input_kmers, model, output):

    inputset = input_fasta_fastq
    type_ = type_
    k_mers = input_kmers
    modelPath = model
    resultPath = output
    if resultPath[-1]!='/':
        resultPath = resultPath+'/'

    logger = logging.getLogger(f"amaisepro")
    logger.setLevel(logging.DEBUG)
    logging.captureWarnings(True)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    consoleHeader = logging.StreamHandler()
    consoleHeader.setFormatter(formatter)
    consoleHeader.setLevel(logging.INFO)
    logger.addHandler(consoleHeader)

    fileHandler = logging.FileHandler(f"{resultPath}info.log")
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    # logger.info(f"Model path: {modelPath}")
    # logger.info(f"Input path: {inputset}")
    # logger.info(f"Labels path: {labelset}")
    # logger.info(f"Results path: {resultPath}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Device: {device}")

    k_mer_arr = pd.read_csv(k_mers, header=None).to_numpy()

    input_data = []

    logger.info("parsing data...")

    startTime = time.time()

    i = 0
    for row in k_mer_arr:
        input_data.append(np.reshape(row.astype(np.float32), (-1, 1)))
        i = i + 1

    accession_numbers = []
    for seq in SeqIO.parse(inputset, type_):
        accession_numbers.append(seq.id)

    endTime = time.time()
    encoding_time_diff = (endTime - startTime) / 60
    logger.info(f"Total time taken to parse data: {encoding_time_diff} min")

    logger.info("initializing and loading the model...")
    model = nn.DataParallel(TCN())
    model.load_state_dict(torch.load(modelPath, device))
    model = model.to(device)
    model.eval()

    logger.info("predicting the classes...")

    startTime_ = time.time()
    dataLoader = DataLoader(input_data, shuffle=False, batch_size=2048)
    # input_data = input_data.to(device)
    predicted = []
    with torch.no_grad():
        for step, test_x in enumerate(dataLoader):
            test_x = test_x.to(device)
            pred = torch.nn.functional.softmax(model(test_x), dim=1)
            _, predicted_labels = torch.max(pred, 1)
            predicted.extend(predicted_labels.cpu().numpy())


    endTime = time.time()
    predicting_time_diff = (endTime - startTime_) / 60
    logger.info(f"Time taken to predict the results: {predicting_time_diff} min")

    pred_df = pd.DataFrame(
        {"id": accession_numbers, "pred_label": predicted}
    )
    pred_df.to_csv(f"{resultPath}predictions.csv", index=False)

    id_label_dict = dict(zip(pred_df['id'], pred_df['pred_label']))
    class_seqs = [[],[],[],[],[],[]]
    for seq in SeqIO.parse(inputset, type_):
        class_seqs[id_label_dict[seq.id]].append(seq)

    class_names = ['host','bacteria','virus','fungi','archaea','protozoa']
    for i in range(1, 6):
        with open(f"{resultPath}{class_names[i]}.{type_}", "w") as file:
            SeqIO.write(class_seqs[i], file, type_)

    endTime = time.time()
    memory = psutil.Process().memory_info()

    logger.info(
        "Total time: {:.2f} min".format(
            (endTime - startTime) / 60
        )
    )
    logger.info(f"Memory usage: {memory}")


if __name__ == "__main__":
    main()
