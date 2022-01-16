import csv
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as prf


preds = []
golds = []

for i in range(5):
    pred_path = f'models/model_artifact_function/{i}/predicted_test.csv'
    gold = [x[2] for x in csv.reader(open(pred_path))][1:]
    pred = [x[3] for x in csv.reader(open(pred_path))][1:]
    assert len(pred) == len(gold)
    preds += pred
    golds += gold

print('Accuracy: {:.1f}'.format(accuracy_score(golds, preds)*100))
macro = np.array(prf(golds, preds, labels=list(range(43)), average='macro', zero_division=0)[:3])*100
print('Precision, Recall, and F1: {:.1f} {:.1f} {:.1f}'.format(*macro))
