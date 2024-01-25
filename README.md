# AMAISE_PRO

cd Documents/codes/AMAISE_PRO
go to ErrorTesting branch

## for original training

python3 train.py -m **modelNameWithPath** -i **trainsetWithPath** -l **labelsetWithPath**

 or

python3 train_new.py -m **modelNameWithPath** -i **trainsetWithPath** -l **labelsetWithPath**

ex: 

python3 train.py -m ../../TestResults/human_test/human_test_model -i ../../Metagenome/human/human_metagenome.fasta -l ../../Metagenome/human/human_metagenome.csv


## Transfer Learning

python3 tltrain.py -m **existingModelNameWithPath** -i **trainsetWithPath** -l **labelsetWithPath** -n **newModelNameWithPath**

ex:

python3 tltrain.py -m models_and_references/single_end_model -i ../../Metagenome/shark/train/shark_metagenome.fasta -l ../../Metagenome/shark/train/shark_metagenome.csv -n ../../TestResults/human_test_tl/human_test_tl_model

## for testing

python3 host_depletion.py -i **inputfile** -t **typefile** -o **outfolder** -m **model**

ex:

python3 host_depletion.py -i ../../Metagenome/human/human_metagenome.fasta -t fasta -o ../../TestResults/human_test/train -m ../../TestResults/human_test/human_test_model

## for evaluating

python3 evaluation.py -o **fileToWrite** -t **truefile** -p **predfile**

ex:

python3 evaluation.py -o ../../TestResults/human_test/train/eval_sum.txt -t ../../Metagenome/human/human_metagenome.csv -p ../../TestResults/human_test/train/mlprobs.txt


