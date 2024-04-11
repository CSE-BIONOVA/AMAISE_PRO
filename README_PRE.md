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

### for classifying sequences using AMAISE_Pro

```sh
python3 mclass_classification.py -i **pathToInputFile** -t **typeOfInputFile** -k **pathToEncodedInputFile** -m **pathToUsedModel\(optional\)** -o **pathToOutputFolder**
```
1) Arguments:

- **i : pathToInputFile** = path to input data (reads in a fasta or fastq file)
- **t : typeOfInputFile** = type of the input data file (fasta or fastq)
- **k : pathToEncodedInputFile** = path to generated 3-mers file (.csv)
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

### for evaluating results

```sh
python3 evaluation.py -p **pathToPredFile** -t **pathToTrueFile**
```
1) Arguments:

- **p : pathToPredFile** = path to generated file of predicted labels (.csv)
- **t : pathToTrueFile** = path to file of true labels (csv file with two columns: accesion codes and corresponding true labels) 
*0 for host, 1 for bacteria, 2 for virus, 3 for fungi, 4 for archaea, and 5 for protozoa*

### for retraining with a different host

```sh
python3 re_train.py -i ../../seq2vecs/human_train_final/3mers -l ../../../shared/human/train/human_train_final_labels.csv -m ../../final_results/models/final_human/amaisepro_3mers_b256_e300_lr0.001 -p **pathTo** -o ../../final_results/logs/amaisepro_seq2vecs_3mers_b256_e300_lr0.001 -b 256 -e 300 -lr 0.001
 ```   

