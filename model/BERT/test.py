import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

from allennlp.modules.seq2seq_encoders import (
    Seq2SeqEncoder,
    PassThroughEncoder,
    LstmSeq2SeqEncoder,
)

def torch_topk_test():
    probs = torch.tensor(
        [[0.1,0.2,0.3],
         [0.2,0.5,0.9]])
    print(probs.shape)
    t = torch.topk(probs,2,dim=0)
    print(t)
    print(t.indices)
    print(t.values)

    # t = torch.topk(probs,3,dim=1)
    # print(t)
    # print(t.indices)
    # print(t.values)

metric = {'NLU': {},
          'AP': {},
          'RG': {}}

y_pred_intent = np.array([0,0,1,1,0,1,0,0])
y_gt_intent = np.array([0,0,0,0,0,1,0,0])
metric['NLU']['intent_macro'] = f1_score(y_gt_intent, y_pred_intent, average='macro')
metric['NLU']['intent_micro'] = f1_score(y_gt_intent, y_pred_intent, average='micro')
print(metric)