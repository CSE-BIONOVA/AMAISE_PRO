import torch
import pandas as pd
from torch.utils.data import DataLoader
from helper import *
import torch.nn as nn
import sys, getopt
from sklearn.metrics import classification_report
try:
    opts, args = getopt.getopt(sys.argv[1:], 'm:i:l:o:')
except getopt.GetoptError:
    sys.exit(2)
    
for opt, arg in opts:
    if opt == '-h':
        sys.exit()
    elif opt in ("-m", "--model"):
        modelpath = arg
    elif opt in ("-i", "--inputfile"):
        inputset = arg
    elif opt in ("-l", "--labelfile"):
        labelset = arg
    elif opt in ("-o", "--outfolder"):
        result_path = arg

BATCH_SIZE = 2048
model = TCN()
model = nn.DataParallel(model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.load_state_dict(torch.load(modelpath,device))
model.to(device)
model.eval()

label_df = pd.read_csv(labelset)
data_arr = pd.read_csv(inputset, header=None).to_numpy()

def encodeLabel(num):
    encoded_l = np.zeros(6)
    encoded_l[num-1] = 1
    # print(num, encoded_l)
    return encoded_l
i=0
# X = []
# y = []
test_data = []
# for seq in SeqIO.parse(inputset, "fasta"):
#     add_len = 9000
#     encoded = generate_long_sequences(seq+"0"*add_len)[:add_len]
#     label = encodeLabel(label_df[i][1])
#     test_data.append((encoded, label))
#     # X.append(encoded)
#     # y.append(label)

#     i+=1

for row in data_arr:
    test_data.append((np.reshape(row.astype(np.float32), (-1, 1)), encodeLabel(label_df['y_true'][i])))
    i = i + 1

file = open(result_path, 'a')
testDataLoader = DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE)
correct_test_predictions = 0
predicted = []
true = []
with torch.no_grad():
    for step, (test_x, test_y) in enumerate(testDataLoader):
        test_x, test_y = test_x.clone().detach().float().to(device), test_y.to(device)

        pred = torch.softmax(model(test_x), dim=1)

        _, predicted_labels = torch.max(pred, 1)
        _, true_labels = torch.max(test_y, 1)
        predicted.extend(predicted_labels.cpu())
        true.extend(true_labels.cpu())
        correct_test_predictions += (predicted_labels == true_labels).sum().item()

test_accuracy = correct_test_predictions / len(testDataLoader.dataset)
print("Accuracy: ", test_accuracy)
file.write("Accuracy: {:.2f}\n".format(test_accuracy))

print(classification_report(true, predicted, target_names = ['Host','Bacteria','Virus','Fungi','Archaea','Protozoa']))
file.write(classification_report(true, predicted, target_names = ['Host','Bacteria','Virus','Fungi','Archaea','Protozoa']))
file.close()
