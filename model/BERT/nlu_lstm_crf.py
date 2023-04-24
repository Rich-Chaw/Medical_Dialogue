#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/4/15 17:17
# @Author : kzl
# @Site :
# @File : basline.py
from allennlp.data import (
    DataLoader,
    DatasetReader,
    Instance,
    Vocabulary,
    TextFieldTensors,
)
from allennlp.modules.seq2seq_encoders import (
    Seq2SeqEncoder,
    PassThroughEncoder,
    LstmSeq2SeqEncoder,
    StackedBidirectionalLstmSeq2SeqEncoder
)


from allennlp.models.archival import archive_model, load_archive
import os
from typing import List, Dict, Tuple, Iterable
import tempfile
import torch
from allennlp.data.data_loaders import SimpleDataLoader
from overrides import overrides
import numpy as np
from allennlp.common.params import Params
from allennlp.commands.train import train_model
from allennlp.data import Instance
from allennlp.data.fields import TextField, MultiLabelField, ListField, Field, MetadataField ,LabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer,PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, Tokenizer,PretrainedTransformerTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules import (
    ConditionalRandomField,
    FeedForward,
)
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.matrix_attention import BilinearMatrixAttention
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders import seq2vec_encoder
from allennlp.nn.util import get_text_field_mask, masked_softmax
from allennlp.common.util import JsonDict
from allennlp.training.metrics import F1Measure, Average, Metric, CategoricalAccuracy
from allennlp.predictors import Predictor
#from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
#from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
#from allennlp.training.trainer import GradientDescentTrainer, Trainer
from allennlp.training.optimizers import AdamOptimizer
from allennlp.nn.regularizers import RegularizerApplicator
from allennlp.data.tokenizers import Tokenizer, WhitespaceTokenizer, CharacterTokenizer
import torch.nn.functional as F
#from DataRead import *
from allennlp.nn import util
from allennlp.nn.util import get_text_field_mask, masked_softmax, dist_reduce_sum
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import re
from allennlp.common.util import int_to_device, is_distributed
from allennlp.common.checks import check_dimensions_match, ConfigurationError
import torch.distributed as dist
import pickle
from typing import Dict, Optional, List, Any, cast

class Average_tensor_origin(Metric):
    """
    This [`Metric`](./metric.md) breaks with the typical `Metric` API and just stores values that were
    computed in some fashion outside of a `Metric`.  If you have some external code that computes
    the metric for you, for instance, you can use this to report the average result using our
    `Metric` API.
    """
    def __init__(self, dim: int) -> None:
        self._total_value = torch.zeros([10,dim], dtype=torch.float)
        self._count = 0

    @overrides
    def __call__(self, value: torch.Tensor):
        """
        # Parameters

        value : `float`
            The value to average.
        """
        v_shape = value.shape
        if v_shape[0] != 10:
            pad = 10-v_shape[0]
            pad_tensor = torch.zeros(pad,v_shape[1])
            value = torch.cat((value,pad_tensor))
        d_tensor = self.detach_tensors(value.float())
        list_tensor = list(d_tensor)
        self._total_value += dist_reduce_sum(list_tensor[0])
        # self._total_value += list_tensor[0]
        self._count += dist_reduce_sum(1)

    @overrides
    def get_metric(self, reset: bool = False):
        """
        # Returns

        The average of all values that were passed to `__call__`.
        """

        average_value = self._total_value / self._count if self._count > 0 else 0.0
        if reset:
            self.reset()
        return average_value

    @overrides
    def reset(self):
        self._total_value = 0.0
        self._count = 0

class Average_tensor(Metric):
    """
    This [`Metric`](./metric.md) breaks with the typical `Metric` API and just stores values that were
    computed in some fashion outside of a `Metric`.  If you have some external code that computes
    the metric for you, for instance, you can use this to report the average result using our
    `Metric` API.
    """

    def __init__(self, dim: int) -> None:
        self._total_value = torch.zeros([10, dim], dtype=torch.float)
        self._count = 0

    @overrides
    def __call__(self, value: torch.Tensor):
        """
        # Parameters

        value : `float`
            The value to average.
        """
        v_shape = value.shape
        if v_shape[0] != 10:
            pad = 10 - v_shape[0]
            pad_tensor = torch.zeros(pad, v_shape[1])
            value = torch.cat((value, pad_tensor))
        
        d_tensor = self.detach_tensors(value.float())
        list_tensor = list(d_tensor)

        if not is_distributed():
            self._total_value += list_tensor[0]
            self._count += 1
        else:
            value_tensor = list_tensor[0]
            dist.all_reduce(value_tensor, dist.ReduceOp.SUM)
            self._total_value += value_tensor
            self._count += dist_reduce_sum(1)


    @overrides
    def get_metric(self, reset: bool = False):
        """
        # Returns

        The average of all values that were passed to `__call__`.
        """

        average_value = self._total_value / self._count if self._count > 0 else 0.0
        if reset:
            self.reset()
        return average_value

    @overrides
    def reset(self):
        dim = (self._total_value.shape)[1]
        self._total_value = torch.zeros([10, dim], dtype=torch.float)
        self._count = 0

###
intent1 = {'Inform': 0, 'Inquire': 1, 'QuestionAnswering': 2, 'Other': 3, 'Chitchat': 4}    # 5 intents1
slot = ['disease', 'symptom', 'treatment', 'other', 'department', 'time', 'precaution', 'medicine', 'pathogeny',
        'side_effect',
        'effect', 'temperature', 'range_body', 'degree', 'frequency', 'dose', 'check_item', 'medicine_category',
        'medical_place', 'disease_history']     #20 slots
intent_slot1 = {}
index = 0
for x in ['Inform', 'Inquire']:
    for y in slot:
        intent_slot1['' + x + ' ' + y] = index
        index += 1
intent_slot1.update({'Inform': 40, 'Inquire': 41, 'QuestionAnswering': 42, 'Other': 43, 'Chitchat': 44})
'''
all seq2vec baseline
'''
def convert(tensor, vocab):
    # 把 tensor中只关于intent的部分提取出来
    result = torch.zeros(tensor.size(0), 5).to('cuda')
    for count, x in enumerate(tensor):
        for id, y in enumerate(x):
            if y == 1.:
                label = vocab.get_token_from_index(id, namespace='labels')
                if label != '' and label.split(' ')[0] in intent1.keys():
                        result[count][int(intent1[label.split(' ')[0]])] = 1.
    return result

def indices(tensor): # get same label
    #result = torch.zeros(tensor.size(0), 45).to('cuda')
    # t1 = tensor[:, 0:22]
    # t2 = tensor[:, 23:26]
    # t3 = tensor[:, 27].unsqueeze(1)
    # t4 = tensor[:, 29:35]
    # t5 = tensor[:, 36:38]
    # t6 = tensor[:, 40:43]
    # t7 = tensor[:, 46].unsqueeze(1)
    # t8 = tensor[:, 51:54]
    # t9 = tensor[:, 55:57]
    # t10 = tensor[:, 60].unsqueeze(1)
    # t11 = tensor[:, 66].unsqueeze(1)
    # return torch.cat([t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11], 1)
    t1 = tensor[:, 0:22]
    t2 = tensor[:, 23:26]
    t3 = tensor[:, 27].unsqueeze(1)
    t4 = tensor[:, 29:35]
    t5 = tensor[:, 36:38]
    t6 = tensor[:, 40:43]
    t7 = tensor[:, 44].unsqueeze(1)
    return torch.cat([t1, t2, t3, t4, t5, t6, t7], 1)
import json
import copy
topic_num = 64

@DatasetReader.register("mds_reader")
class TextClassificationTxtReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 model: str = None,
                 max_tokens : int = None) -> None:

        super().__init__()
        self.tokenizer = tokenizer or CharacterTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.model = model
        self.max_tokens = max_tokens

    @overrides
    def _read(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
            # split train data and val data
            #start_index = 0 if 'val' in file_path else int(0.13 * len(lines))
            #end_index = int(0.13 * len(lines)) if 'val' in file_path else -1
            start_index = 0
            end_index = -1
            # add Separator
            Separator = 'intent'
            #Separator = 'intent' if "intent" in file_path else 'action'
            for line in lines[start_index:end_index]:
                if line != '':
                    #text = '<|endoftext|> <|context|> <|endofcontext|> '+line.strip().split('<|'+Separator+'|>')[0].split('<|endofcontext|>')[1]
                    text = line.strip().split('<|' + Separator + '|>')[0]
                    intent = re.sub('[\u4e00-\u9fa5]', '', line.strip().split('<|endof'+Separator+'|>')[0].split('<|'+Separator+'|>')[1])
                    intent_slot = [x.strip() for x in intent.split('<|continue|>') if x.strip() in intent_slot1.keys()]
                    tokens = self.tokenizer.tokenize(text)
                    if self.max_tokens:
                        tokens = tokens[-1*self.max_tokens:]
                    if len(intent_slot) != 0 :
                        text_field = TextField(tokens, self.token_indexers)
                        #label_intent_field = MultiLabelField(intent_list)
                        label_intent_slot_field = MultiLabelField(intent_slot)
                        #print("label_intent_field:", label_intent_field)
                        #print("label_intent_slot_field:", label_intent_slot_field)
                        fields = {'text': text_field, 'label': label_intent_slot_field}
                        yield Instance(fields)

@Model.register("simple_classifier")
class SimpleClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder1: Seq2SeqEncoder,
                 encoder2: Seq2VecEncoder,
                 dropout: float = None,
                 label_encoding: Optional[str] = None,
                 include_start_end_transitions: bool = True,
                 constrain_crf_decoding: bool = None,
                 top_k: int = 1,):
        super().__init__(vocab)
        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None
        
        self.vocab = vocab
        self.num_labels_intent = vocab.get_vocab_size("labels")
        for x in range(self.num_labels_intent):
            print(str(x)+' ', vocab.get_token_from_index(x, namespace='labels'))
        print(self.num_labels_intent) # 45

        if constrain_crf_decoding:
            if not label_encoding:
                raise ConfigurationError(
                    "constrain_crf_decoding is True, but no label_encoding was specified."
                )
            labels = self.vocab.get_index_to_token_vocabulary("lables")
            constraints = allowed_transitions(label_encoding, labels)
        else:
            constraints = None
        
        self.embedder = embedder
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.crf = ConditionalRandomField(
                self.num_labels_intent,
                constraints,
                include_start_end_transitions,
            )
        #vocab = vocab.from_files('tmp/lstm_nlu_P2/vocabulary/')
        '''vocab.add_tokens_to_namespace(['<|endoftext|>', '<|user|>', '<|intent|>', '<|endofintent|>',
                                            '<|action|>', '<|endofaction|>', '<|response|>', '<|endofresponse|>',
                                            'Inform', 'Inquire', 'Recommend', 'Diagnosis', 'Chitchat', 'Other',
                                            'disease',
                                            'symptom', 'treatment', 'other', 'department', 'time', 'precaution',
                                            'QuestionAnswering',
                                            'medicine', 'pathogeny', 'side_effect', 'effect', 'temperature',
                                            'range_body', 'degree',
                                            'frequency', 'dose', 'check_item', 'medicine_category', 'medical_place',
                                            'disease_history',
                                            '<|context|>', '<|endofcontext|>', '<|system|>', '<|currentuser|>',
                                            '<|continue|>', '<|endofcurrentuser|>'], 'tokens')'''
        
        #print('26954', vocab.get_token_from_index(26954, namespace='tokens'))
        self.classifier = torch.nn.Linear(encoder2.get_output_dim(), self.num_labels_intent)
        
        self.include_start_end_transitions = include_start_end_transitions
        self.top_k = top_k
        
        # self.all_pre_intent = Average_tensor(5)
        # self.all_true_intent = Average_tensor(5)
        # self.all_pre_intent_slot = Average_tensor(45)
        # self.all_true_intent_slot = Average_tensor(45)

        batch_size = encoder2.get_output_dim()
        self.all_pre_intent = torch.zeros([batch_size,5])
        self.all_true_intent = torch.zeros([batch_size,5])
        self.all_pre_intent_slot = torch.zeros([batch_size,45])
        self.all_true_intent_slot = torch.zeros([batch_size,45])

    def forward(self,text: Dict[str, torch.Tensor],
                    label: torch.Tensor) -> Dict[str, torch.Tensor]:
            # text Shape: (batch_size, num_tokens, embedding_dim)
            embedded_text = self.embedder(text)                 #embedded_text:  (batch_size, num_tokens=512, embedding_dim)
            mask = util.get_text_field_mask(text)               #mask:           (batch_size, num_tokens)
            if self._dropout:
                embedded_text = self._dropout(embedded_text)
            encoded1_text = self.encoder1(embedded_text, mask)    #encoded1_text.shape: (batch_size,num_tokens,hidden_size*numlayers)
            if self._dropout:
                encoded1_text = self._dropout(encoded1_text) 
            # encoded2_text = self.encoder2(encoded1_text, mask)    #encoded2_text.shape: (batch_size,hidden_size*numlayers)
            print("encoded1_text:",encoded1_text.shape)
            # print("encoded2_text:",encoded2_text.shape)
            
            logits = self.classifier(encoded1_text)              #logits.Shape: (batch_size,num_tokens,num_labels)
            
           
            print("logits.shape",logits.shape)

            # probs = torch.sigmoid(logits)
            best_paths = self.crf.viterbi_tags(logits, mask, top_k=self.top_k)

            # Just get the top tags and ignore the scores.
            predicted_tags = cast(List[List[int]], [x[0][0] for x in best_paths])

            # Represent viterbi tags as "class probabilities" that we can
            # feed into the metrics
            class_probabilities = logits * 0.0
            for i, instance_tags in enumerate(predicted_tags):
                for j, tag_id in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = 1
            probs = self.encoder2(class_probabilities, mask)

            # print("text:",text)
            # print("embedded_text:",embedded_text.shape)
            # print("mask:",mask.shape)
            print("encoded1_text:",encoded1_text.shape)
            # print("encoded2_text:",encoded1_text.shape)
            print("logits.shape",logits.shape)
            print("best_paths",best_paths)
            print("probs",probs)
            print("prob shape:",probs.shape)        #torch.Size([10, 45])
            print("lable shape:",label.shape)       #torch.Size([10, 45])



            # probs = indices(probs)                  #torch.Size([768, 38])
            # label = label[:, 0:probs.size(-1)]
            # label = indices(label)                  #torch.Size([10, 38])
            topic_weight = torch.ones_like(label) + label * (label.size()[1]-1)

            # Shape: (1,)
            loss = torch.nn.functional.binary_cross_entropy(probs, label.float(), topic_weight.float())

            pre_index_intent = convert((probs > 0.5).long(), self.vocab)
            pre_index_intent_slot = (probs > 0.5).long()

            label_intent = convert(label, self.vocab)
            
     
            print("pre_index_intent_slot: ",pre_index_intent_slot)          
            print("lable: ",label)              

            # self.all_pre_intent(pre_index_intent.cpu())             #torch.Size([10, 5])
            # self.all_true_intent(label_intent.cpu())                #torch.Size([10, 5])
            # self.all_pre_intent_slot(pre_index_intent_slot.cpu())   #torch.Size([10, 45])
            # self.all_true_intent_slot(label.cpu())                  #torch.Size([10, 45])
            
            self.all_pre_intent =  pre_index_intent.cpu()            #torch.Size([10, 5])
            self.all_true_intent = label_intent.cpu()                   #torch.Size([10, 5])
            self.all_pre_intent_slot = pre_index_intent_slot.cpu()      #torch.Size([10, 45])
            self.all_true_intent_slot = label.cpu()                     #torch.Size([10, 45])
            
            output = {"logits": logits, "mask": mask, "probs": probs, "loss": loss}
            return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}

        # all_pre_intent = self.all_pre_intent.get_metric(reset=reset)
        # all_true_intent = self.all_true_intent.get_metric(reset=reset)
        # all_pre_intent_slot = self.all_pre_intent_slot.get_metric(reset=reset)
        # all_true_intent_slot = self.all_true_intent_slot.get_metric(reset=reset)

        all_pre_intent = self.all_pre_intent
        all_true_intent = self.all_true_intent
        all_pre_intent_slot = self.all_pre_intent_slot
        all_true_intent_slot = self.all_true_intent_slot
 

        acc, rec, f1, facc = 0., 0., 0., 0.

        # all_pre_intent = (np.array(torch.round(all_pre_intent))).astype(int)

        if torch.is_tensor(all_true_intent):
            all_true_intent = np.array(all_true_intent)
            all_pre_intent = np.array(all_pre_intent)
            all_true_intent_slot = np.array(all_true_intent_slot)
            all_pre_intent_slot = np.array(all_pre_intent_slot)
        else:
            all_true_intent = np.array(round(all_true_intent)).astype(int)
            all_pre_intent = (np.array(round(all_pre_intent))).astype(int)
            all_true_intent_slot = (np.array(round(all_true_intent_slot))).astype(int)
            all_pre_intent_slot = (np.array(round(all_pre_intent_slot))).astype(int)
            return metrics


        pre_micro_intent = precision_score(all_true_intent, all_pre_intent, average='micro')
        rec_micro_intent = recall_score(all_true_intent, all_pre_intent, average='micro')
        acc_micro_intent = accuracy_score(all_true_intent, all_pre_intent)
        f1_micro_sk_intent = f1_score(all_true_intent, all_pre_intent, average='micro')
        f1_macro_sk_intent = f1_score(all_true_intent, all_pre_intent, average='macro')
        f1_weighted_intent = f1_score(all_true_intent, all_pre_intent, average='weighted')

        metrics['acc_intent'] = acc_micro_intent
        metrics['pre_intent'] = pre_micro_intent
        metrics['rec_intent'] = rec_micro_intent
        metrics['f1_micro_sk_intent'] = f1_micro_sk_intent
        metrics['f1_macro_sk_intent'] = f1_macro_sk_intent
        metrics['f1_weighted_intent'] = f1_weighted_intent

        pre_micro_intent_slot = precision_score(all_true_intent_slot, all_pre_intent_slot, average='micro')
        rec_micro_intent_slot = recall_score(all_true_intent_slot, all_pre_intent_slot, average='micro')
        acc_micro_intent_slot = accuracy_score(all_true_intent_slot, all_pre_intent_slot)
        f1_micro_sk_intent_slot = f1_score(all_true_intent_slot, all_pre_intent_slot, average='micro')
        f1_macro_sk_intent_slot = f1_score(all_true_intent_slot, all_pre_intent_slot, average='macro')
        f1_weighted_intent_slot = f1_score(all_true_intent_slot, all_pre_intent_slot, average='weighted')


        print("all_pre_intent_slot:",all_pre_intent_slot)
        print("all_true_intent_slot:",all_true_intent_slot)
        print("f1_micro_sk_intent_slot:",f1_micro_sk_intent_slot)
        
        metrics['acc_intent_slot'] = acc_micro_intent_slot
        metrics['pre_intent_slot'] = pre_micro_intent_slot
        metrics['rec_intent_slot'] = rec_micro_intent_slot
        metrics['f1_micro_sk_intent_slot'] = f1_micro_sk_intent_slot
        metrics['f1_macro_sk_intent_slot'] = f1_macro_sk_intent_slot
        metrics['f1_weighted_intent_slot'] = f1_weighted_intent_slot

        return metrics
    
def build_data_reader() -> DatasetReader:
    print("building data_reader")
    tokenizer = PretrainedTransformerTokenizer(model_name="hfl/chinese-bert-wwm")
    token_indexer = {"bert": PretrainedTransformerIndexer(model_name="hfl/chinese-bert-wwm")}
    return TextClassificationTxtReader(tokenizer=tokenizer, token_indexers=token_indexer, max_tokens=512)



def evaluate_BERT():
    model_path = "BERT-BASE-CHINESE2-NLU"
    archive = load_archive(os.path.join(model_path,"model.tar.gz"))
    model = archive.model
    dataset_reader = build_data_reader()
    test_data = list(dataset_reader.read("../../data/test_human_annotation.txt"))
    data_loader = SimpleDataLoader(test_data, batch_size=10)
    data_loader.index_with(model.vocab)
    output_dict = model(data_loader)
    iterator = iter(data_loader)
    loss = output_dict.get("loss")
    metrics = model.get_metrics()
    metrics_json = json.dumps(metrics, indent=2)
    file_path = "BERT_metrics"
    if file_path:
        with open(file_path, "w") as metrics_file:
            metrics_file.write(metrics_json)