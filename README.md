# AMAISE_PRO

for testing
python3 host_depletion.py -i <inputfile> -t <typefile> -o <outfolder> -m <model>
model = 'models_and_references/model_new'

python3 host_depletion.py -i ../../Metagenome/shark/test/shark_metagenome.fasta -t fasta -o ../../TestResults/shark_long_tl -m ../../TestResults/shark_long_tl/shark_long_tl_

for training
python3 train.py -m <pathtosavemodel> -i <trainset> -l <labelset>
pathtosavemodel  = 'models_and_references/model_new'
trainset = 'train_data/set1.csv'

python3 tltrain.py -m models_and_references/single_end_model -i Metagenome/shark/train/shark_metagenome.fasta -l ../../Metagenome/shark/train/shark_metagenome.csv -n ../../TestResults/shark_long_tl/shark_long_tl_

for evaluating
python3 evaluation.py -o <fileToWrite> -t <truefile> -p <predfile>
fileToWrite = 'Testings/eval_summary.txt
truefile = 'train_data/human.csv'
predfile = 'Testings/test/mlprobs.txt'


models - models_and_references
model_name - human_short_tl, shark_long_o

