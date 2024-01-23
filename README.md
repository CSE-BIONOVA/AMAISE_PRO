# AMAISE_PRO

cd Documents/codes/AMAISE_PRO
go to ErrorTesting branch

## for original training

python3 train.py -m **modelNameWithPath** -i **trainsetWithPath** -l **labelsetWithPath**

ex: 

python3 train.py -m ../../TestResults/human_long/human_long_test -i ../../Metagenome/shark/train/shark_metagenome.fasta -l ../../Metagenome/shark/train/shark_metagenome.csv

## Transfer Learning

python3 tltrain.py -m **existingModelNameWithPath** -i **trainsetWithPath** -l **labelsetWithPath** -n **newModelNameWithPath**

ex:

python3 tltrain.py -m models_and_references/single_end_model -i ../../Metagenome/shark/train/shark_metagenome.fasta -l ../../Metagenome/shark/train/shark_metagenome.csv -n ../../TestResults/human_long_tl/human_long_tl_test

## for testing

python3 host_depletion.py -i **inputfile** -t **typefile** -o **outfolder** -m **model**

ex:

python3 host_depletion.py -i ../../Metagenome/human/human_metagenome.fasta -t fasta -o ../../TestResults/human_long/train -m ../../TestResults/human_long/human_long_test

## for evaluating

python3 evaluation.py -o **fileToWrite** -t **truefile** -p **predfile**

ex:

python3 evaluation.py -o ../../TestResults/human_long/train/eval_sum.txt -t ../../Metagenome/human/human_metagenome.csv -p ../../TestResults/human_long/train/mlprobs.txt


