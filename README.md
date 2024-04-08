# AMAISE_PRO

When using the server, 
```sh
cd Documents/codes/AMAISE_PRO
```
## Requirements

To install required libraries and packages,

```sh
pip install requirements.txt
```
go to MultiClzImpl branch

## for original training

python3 train.py -m **modelNameWithPath** -i **trainsetWithPath** -l **labelsetWithPath** -o **filenameWithPath**

- **modelNameWithPath** = model name with path to save the model

- **trainsetWithPath** = .fasta file (train set)

- **labelsetWithPath** = .csv file (train set)

- **filenameWithPath** = file name with path to save the results

ex: 

```sh
python3 train.py -m ../Models/human_model -i ../Final_dataset/human_train/human_train.fasta -l ../Final_dataset/human_train/human_train_labels.csv -o ../TestResults/result.txt
```

## Transfer Learning

python3 tltrain.py -m **existingModelNameWithPath** -i **trainsetWithPath** -l **labelsetWithPath** -n **newModelNameWithPath** -o **filenameWithPath**

- **existingModelNameWithPath** = existing model name with path

- **trainsetWithPath** = .fasta file (train set)

- **labelsetWithPath** = .csv file (train set)

- **newModelNameWithPath** = model name with path to save the new model

- **filenameWithPath** = file name with path to save the results

ex:

```sh
python3 tltrain.py -m ../Models/human_model -i .Final_dataset/shark_train/shark_train.fasta -l ../Final_dataset/shark_train/shark_train_labels.csv -n ../Models/shark_model -o ../TestResults/result_tl.txt
```
## for testing

python3 test_model.py -m **existingModelNameWithPath** -i **testsetWithPath** -l **labelsetWithPath** -o **filenameWithPath**

- **modelNameWithPath** = existing model name with path

- **testsetWithPath** = .fasta file (test set)

- **labelsetWithPath** = .csv file (test set)

- **filenameWithPath** = file name with path to save the results

ex: 

```sh
python3 test_model.py -m ../Models/human_model -i ../Final_dataset/human_test/human_test.fasta -l ../Final_dataset/human_test/human_test_labels.csv -o ../TestResults/result_test.txt
```

