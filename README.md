# AMAISE_Pro

AMAISE_Pro is a novel, multi-class classification tool with the host depletion. Given a set of reads, for each sequence, AMAISE outputs a classification label determining what is the superkingdom it belongs to or does it belongs to host (0 for host, 1 for bacteria, 2 for virus, 3 for fungi, 4 for archaea, and 5 for protozoa). AMAISE then stores the sequences of each type in 6 files for downstream analysis.

This contains 
1) 

____________________________________________________________________________________________
## System Requirements

First download this Github Repository,

```sh
git clone https://github.com/CSE-BIONOVA/AMAISE_PRO.git
```

AMAISE requires a computational environment with a Linux-based machine/Ubuntu.

Required packages and versions are listed in "requirements.txt". You can install the packages in requirements.txt using:

```sh
pip install -r requirements.txt
```
____________________________________________________________________________________________
## Usage Notes for AMAISE_Pro

### For classifying sequences using AMAISE_Pro

python3 mclass_classification.py -i **pathToInputFile** -t **typeOfInputFile** -k **pathToEncodedInputFile** -m **pathToUsedModel(optional)** -o **pathToOutputFolder**

1) Arguments:

- **i : pathToInputFile** = path to input data (reads in a fasta or fastq file)
- **t : typeOfInputFile** = type of the input data file (fasta or fastq)
- **k : pathToEncodedInputFile** = path to seq2vec-enocoded input data (.csv)
- **m : pathToUsedModel** = optional (if you want to use other model instead of original AMAISE_Pro, path to the model which is going to be used should be provided here)
- **o : pathToOutputFolder** = path to the folder that you want to put the final results and predictions

2) Outputs (saved into output folder)

- **predictions.csv**: csv file of accession codes and corresponiding predicted labels
- **host.fastq(.fasta)**: fastq or fasta file of classified host sequences
- **bacteria.fastq(.fasta)**: fastq or fasta file of classified bacteria sequences
- **virus.fastq(.fasta)**: fastq or fasta file of classified virus sequences
- **fungi.fastq(.fasta)**: fastq or fasta file of classified fungi sequences
- **archaea.fastq(.fasta)**: fastq or fasta file of classified archaea sequences
- **protozoa.fastq(.fasta)**: fastq or fasta file of classified protozoa sequences

ex:

```sh
python3 mclass_classification.py -i example/test.fasta -t fasta -k example/test_3mers.csv -o example/results
```
### For evaluating results

python3 evaluation.py -p **pathToPredFile** -t **pathToTrueFile**

1) Arguments:

- **p : pathToPredFile** = path to generated file of predicted labels (.csv)
- **t : pathToTrueFile** = path to file of true labels (csv file with two columns: accesion codes and corresponding true labels)
    - *0 for host, 1 for bacteria, 2 for virus, 3 for fungi, 4 for archaea, and 5 for protozoa*
    - *ex: example/test_labels.csv*

2) Outputs - classification report as a terminal output

ex:

```sh
python3 evaluation.py -p example/results/predictions.csv -t example/test_labels.csv
```

### For retraining with a different host

python3 re_train.py -i **pathToTrainData(k-mers)** -l **pathToTrainDataLabels** -m **pathToSaveModel** -p **pathToExistingModel** -o **pathToSaveLogFile** -b **batchSize** -e **#epoches**  -lr **learningRate**

1) Arguments:

- **i : pathToTrainData(k-mers)** = path to train set that is seq2vec-enocoded (.csv)
- **l : pathToTrainDataLabels** = path to labels file of train set (csv file with two columns: accesion codes and corresponding true labels)
    - *ex: example/retrain_labels.csv*


- **m : pathToSaveModel** = path (with model name) to save new model
- **p : pathToExistingModel** = optional (if you want to use other model instead of original AMAISE_Pro as existing model, path to that model should be provided here)
- **o : pathToSaveLogFile** = path (with name for log file) to save log file
- **b : batchSize** = batch size
- **e : #epoches** = number of epoches
- **lr : learningRate** = initial learning rate

2) Outputs

- re-trained model in the provided path
- log file in the provided path

ex:

```sh
python3 re_train.py -i example/retrain_3mers.csv -l example/retrain_labels.csv -m example/retrained_model -o example/retrain_info -b 256 -e 30 -lr 0.001
 ```   

