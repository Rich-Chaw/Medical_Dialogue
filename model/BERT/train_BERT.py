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
from allennlp.data.fields import TextField, MultiLabelField, ListField, Field, MetadataField, LabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import (
    TokenIndexer,
    SingleIdTokenIndexer,
    PretrainedTransformerIndexer,
    PretrainedTransformerMismatchedIndexer,
)
from allennlp.common.tqdm import Tqdm
from allennlp.data.tokenizers import (
    Token,
    Tokenizer,
    PretrainedTransformerTokenizer,
    SpacyTokenizer,
)
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, BertPooler
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.matrix_attention import BilinearMatrixAttention
from allennlp.modules.token_embedders import Embedding,PretrainedTransformerEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask, masked_softmax
from allennlp.common.util import JsonDict
from allennlp.training.metrics import F1Measure, Average, Metric, CategoricalAccuracy
from allennlp.predictors import Predictor
# from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
# from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
# from allennlp.training.trainer import GradientDescentTrainer, Trainer
from allennlp.training.optimizers import AdamOptimizer,HuggingfaceAdamWOptimizer
from allennlp.nn.regularizers import RegularizerApplicator
from allennlp.data.tokenizers import Tokenizer, WhitespaceTokenizer, CharacterTokenizer
import torch.nn.functional as F
# from DataRead import *
from allennlp.nn import util
from allennlp.training.trainer import Trainer, GradientDescentTrainer
from allennlp.nn.util import get_text_field_mask, masked_softmax, dist_reduce_sum
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import re
from allennlp.training.util import evaluate
import pickle
import json
import copy
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder

### initialize the slot
intent1 = {'Inform': 0, 'Inquire': 1, 'QuestionAnswering': 2, 'Other': 3, 'Chitchat': 4}  # 5 intents1
slot = ['disease', 'symptom', 'treatment', 'other', 'department', 'time', 'precaution', 'medicine', 'pathogeny',
        'side_effect',
        'effect', 'temperature', 'range_body', 'degree', 'frequency', 'dose', 'check_item', 'medicine_category',
        'medical_place', 'disease_history']  # 20 slots
intent_slot1 = {}
index = 0
for x in ['Inform', 'Inquire']:
    for y in slot:
        intent_slot1['' + x + ' ' + y] = index
        index += 1
intent_slot1.update({'Inform': 40, 'Inquire': 41, 'QuestionAnswering': 42, 'Other': 43, 'Chitchat': 44})
topic_num = 64


'''
all seq2vec baseline
'''

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
        # self._total_value += dist_reduce_sum(list_tensor[0])
        self._total_value += list_tensor[0]
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

def indices(tensor):  # get same label
    # result = torch.zeros(tensor.size(0), 45).to('cuda')
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


@DatasetReader.register("mds_reader")
class TextClassificationTxtReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 model: str = None,
                 max_tokens: int = None) -> None:

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
            # start_index = 0 if 'val' in file_path else int(0.13 * len(lines))
            # end_index = int(0.13 * len(lines)) if 'val' in file_path else -1
            start_index = 0
            end_index = -1
            # add Separator
            Separator = 'intent'
            # Separator = 'intent' if "intent" in file_path else 'action'
            for line in lines[start_index:end_index]:
                if line != '':
                    # text = '<|endoftext|> <|context|> <|endofcontext|> '+line.strip().split('<|'+Separator+'|>')[0].split('<|endofcontext|>')[1]
                    text = line.strip().split('<|' + Separator + '|>')[0]
                    intent = re.sub('[\u4e00-\u9fa5]', '',
                                    line.strip().split('<|endof' + Separator + '|>')[0].split('<|' + Separator + '|>')[
                                        1])
                    intent_slot = [x.strip() for x in intent.split('<|continue|>') if x.strip() in intent_slot1.keys()]
                    tokens = self.tokenizer.tokenize(text)
                    if self.max_tokens:
                        tokens = tokens[-1 * self.max_tokens:]
                    if len(intent_slot) != 0:
                        text_field = TextField(tokens, self.token_indexers)
                        # label_intent_field = MultiLabelField(intent_list)
                        label_intent_slot_field = MultiLabelField(intent_slot)
                        # print("label_intent_field:", label_intent_field)
                        # print("label_intent_slot_field:", label_intent_slot_field)
                        fields = {'text': text_field, 'label': label_intent_slot_field}
                        yield Instance(fields)


@Model.register("simple_classifier")
class SimpleClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 dropout: float = None):
        super().__init__(vocab)
        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None
        self.embedder = embedder
        self.encoder = encoder
        # vocab = vocab.from_files('tmp/lstm_nlu_P2/vocabulary/')
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
        num_labels_intent = vocab.get_vocab_size("labels")
        for x in range(num_labels_intent):
            print(str(x) + ' ', vocab.get_token_from_index(x, namespace='labels'))
        print(num_labels_intent)  # 45
        # print('26954', vocab.get_token_from_index(26954, namespace='tokens'))
        # print("encoder.get_output_dim()",encoder.get_output_dim()) == 10
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels_intent)
        self.vocab = vocab
        ''' self.total_pre = Average()
        self.total_true = Average()
        self.total_pre_true = Average()
        self.accuracy = Average()

        self.total_pre_macro = Average()
        self.total_true_macro = Average()
        self.total_pre_true_macro = Average()'''

        self.all_pre_intent = Average_tensor(5)
        self.all_true_intent = Average_tensor(5)
        self.all_pre_intent_slot = Average_tensor(45)
        self.all_true_intent_slot = Average_tensor(45)

        '''self.total_pre_macro = torch.zeros(num_labels)
        self.total_true_macro = torch.zeros(num_labels)
        self.total_pre_true_macro = torch.zeros(num_labels)
        self.all_pre = torch.zeros(num_labels)
        self.all_true = torch.zeros(num_labels)'''
        # self.macro_f = MacroF(num_labels)

    def forward(self, text: Dict[str, torch.Tensor],
                label: torch.Tensor) -> Dict[str, torch.Tensor]:
        # text Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)  # embedded_text:  (batch_size, num_tokens, embedding_dim)
        mask = util.get_text_field_mask(text)  # mask:           (batch_size, num_tokens)
        encoded_text = self.encoder(embedded_text, mask)  # encoded_text.shape: (batch_size,embedding_dim)
        logits = self.classifier(encoded_text)  # logits.Shape: (batch_size, num_labels)
        probs = torch.sigmoid(logits)

        print("text:", text)
        print("embedded_text:", embedded_text)
        print("mask:", mask.shape)
        print("encoded_text:", encoded_text)
        print("prob shape:", probs)  # torch.Size([10, 45])
        print("lable shape:", label)  # torch.Size([10, 45])

        # probs = indices(probs)                  #torch.Size([768, 38])
        # label = label[:, 0:probs.size(-1)]
        # label = indices(label)                  #torch.Size([10, 38])
        topic_weight = torch.ones_like(label) + label * (label.size()[1] - 1)

        # Shape: (1,)
        loss = torch.nn.functional.binary_cross_entropy(probs, label.float(), topic_weight.float())

        pre_index_intent = convert((probs > 0.5).long(), self.vocab)
        pre_index_intent_slot = (probs > 0.5).long()

        label_intent = convert(label, self.vocab)

        # self.macro_f(pre_index.cpu(), label.cpu())
        '''total_pre = torch.sum(pre_index)
        total_true = torch.sum(label)
        mask_index = (label == 1).long()
        true_positive = (pre_index == label).long() * mask_index
        pre_true = torch.sum(true_positive)'''

        '''self.total_pre(total_pre.float().item())
        self.total_true(total_true.float().item())
        self.total_pre_true(pre_true.float().item())'''

        '''self.total_pre_macro(torch.sum(pre_index, dim=0))
        self.total_true_macro(torch.sum(label, dim=0))
        self.total_pre_true_macro(torch.sum(true_positive, dim=0))'''

        print("pre_index_intent ", pre_index_intent)
        print("label_intent", label_intent)
        print("pre_index_intent_slot: ", pre_index_intent_slot)
        print("lable.shape: ", label)

        self.all_pre_intent(pre_index_intent.cpu())  # torch.Size([10, 5])
        self.all_true_intent(label_intent.cpu())  # torch.Size([10, 5])
        self.all_pre_intent_slot(pre_index_intent_slot.cpu())  # torch.Size([10, 45])
        self.all_true_intent_slot(label.cpu())  # torch.Size([10, 45])

        return {'loss': loss, 'probs': probs}

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        '''total_pre = self.total_pre.get_metric(reset=reset)
        total_pre_true = self.total_pre_true.get_metric(reset=reset)
        total_true = self.total_true.get_metric(reset=reset)'''

        '''total_pre_macro = self.total_pre_macro.get_metric(reset=reset)
        total_pre_true_macro = self.total_pre_true_macro.get_metric(reset=reset)
        total_true_macro = self.total_true_macro.get_metric(reset=reset)'''
        all_pre_intent = self.all_pre_intent.get_metric(reset=reset)
        all_true_intent = self.all_true_intent.get_metric(reset=reset)
        all_pre_intent_slot = self.all_pre_intent_slot.get_metric(reset=reset)
        all_true_intent_slot = self.all_true_intent_slot.get_metric(reset=reset)

        # print("all_pre_intent = ",all_pre_intent)
        # print("all_true_intent = ",all_true_intent)

        acc, rec, f1, facc = 0., 0., 0., 0.
        '''acc_macro = torch.zeros(total_pre_macro.size(0))
        rec_macro = torch.zeros(total_pre_macro.size(0))
        f1_macro_temp = torch.zeros(total_pre_macro.size(0))'''
        f1_macro = 0.
        '''if total_pre > 0:
            acc = total_pre_true / total_pre
        if total_true > 0:
            rec = total_pre_true / total_true
        if acc + rec > 0:
            f1 = 2 * acc * rec / (acc + rec)'''

        '''for x in range(total_pre_macro.size(0)):
            if total_pre_macro[x] == 0:
                acc_macro[x] = 0
            else:
                acc_macro[x] = total_pre_true_macro[x] / total_pre_macro[x]

        for x in range(total_pre_macro.size(0)):
            if total_true_macro[x] == 0:
                rec_macro[x] = 0
            else:
                rec_macro[x] = total_pre_true_macro[x] / total_true_macro[x]

        for x in range(total_pre_macro.size(0)):
            if acc_macro[x] + rec_macro[x] == 0:
                f1_macro_temp[x] = 0
            else:
                f1_macro_temp[x] = 2 * acc_macro[x] * rec_macro[x] / (acc_macro[x] + rec_macro[x])
        print("f1_macro_all:", f1_macro_temp)
        f1_macro = float(torch.sum(f1_macro_temp)/total_pre_macro.size(0))'''

        # print("all_true:", all_true)
        # print("all_pre:", all_pre)

        all_true_intent = all_true_intent.view(-1).int()
        all_pre_intent = all_pre_intent.view(-1).int()
        all_true_intent_slot = all_true_intent_slot.view(-1).int()
        all_pre_intent_slot = all_pre_intent_slot.view(-1).int()

        pre_micro_intent = precision_score(all_true_intent, all_pre_intent, average='micro')
        rec_micro_intent = recall_score(all_true_intent, all_pre_intent, average='micro')
        # acc_micro_intent = accuracy_score(all_true_intent, all_pre_intent)
        f1_micro_sk_intent = f1_score(all_true_intent, all_pre_intent, average='micro')
        f1_macro_sk_intent = f1_score(all_true_intent, all_pre_intent, average='macro')
        f1_weighted_intent = f1_score(all_true_intent, all_pre_intent, average='weighted')

        # metrics['acc_intent'] = acc_micro_intent
        metrics['pre_intent'] = pre_micro_intent
        metrics['rec_intent'] = rec_micro_intent

        # metrics['f1'] = f1
        # metrics['acc_macro'] = acc_macro
        # metrics['rec_macro'] = rec_macro
        # metrics['f1_macro'] = f1_macro
        metrics['f1_micro_sk_intent'] = f1_micro_sk_intent
        metrics['f1_macro_sk_intent'] = f1_macro_sk_intent
        metrics['f1_weighted_intent'] = f1_weighted_intent

        pre_micro_intent_slot = precision_score(all_true_intent_slot, all_pre_intent_slot, average='micro')
        rec_micro_intent_slot = recall_score(all_true_intent_slot, all_pre_intent_slot, average='micro')
        # acc_micro_intent_slot = accuracy_score(all_true_intent_slot, all_pre_intent_slot)
        f1_micro_sk_intent_slot = f1_score(all_true_intent_slot, all_pre_intent_slot, average='micro')
        f1_macro_sk_intent_slot = f1_score(all_true_intent_slot, all_pre_intent_slot, average='macro')
        f1_weighted_intent_slot = f1_score(all_true_intent_slot, all_pre_intent_slot, average='weighted')

        # metrics['acc_intent_slot'] = acc_micro_intent_slot
        metrics['pre_intent_slot'] = pre_micro_intent_slot
        metrics['rec_intent_slot'] = rec_micro_intent_slot

        # metrics['f1'] = f1
        # metrics['acc_macro'] = acc_macro
        # metrics['rec_macro'] = rec_macro
        # metrics['f1_macro'] = f1_macro
        metrics['f1_micro_sk_intent_slot'] = f1_micro_sk_intent_slot
        metrics['f1_macro_sk_intent_slot'] = f1_macro_sk_intent_slot
        metrics['f1_weighted_intent_slot'] = f1_weighted_intent_slot

        return metrics

def build_data_reader() -> DatasetReader:
    print("building data_reader")
    tokenizer = PretrainedTransformerTokenizer(model_name="hfl/chinese-bert-wwm")
    token_indexer = {"bert": PretrainedTransformerIndexer(model_name="hfl/chinese-bert-wwm")}
    return TextClassificationTxtReader(tokenizer=tokenizer, token_indexers=token_indexer, max_tokens=512)


def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    print("Building the vocabulary")
    return Vocabulary.from_instances(instances)


def build_model(vocab: Vocabulary) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedder = BasicTextFieldEmbedder(
        token_embedders={"bert": PretrainedTransformerEmbedder(model_name="hfl/chinese-bert-wwm")}
    )
    encoder = BagOfEmbeddingsEncoder(embedding_dim=768)
    # encoder = BertPooler(pretrained_model="hfl/chinese-bert-wwm")
    return SimpleClassifier(vocab, embedder, encoder)


def build_data_loaders(
    train_data: List[Instance],
    dev_data: List[Instance],
) -> Tuple[DataLoader, DataLoader]:
    train_loader = SimpleDataLoader(train_data, batch_size=10, shuffle=True)
    dev_loader = SimpleDataLoader(dev_data, batch_size=10, shuffle=False)
    return train_loader, dev_loader


def build_trainer(
    model: Model,
    serialization_dir: str,
    train_loader: DataLoader,
    dev_loader: DataLoader,
) -> Trainer:
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = HuggingfaceAdamWOptimizer(model_parameters=parameters,lr=2e-5)
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        validation_metric="+f1_micro_sk_intent_slot",
        patience=10,
        num_epochs=30,
        grad_norm=20.0,
        optimizer=optimizer,
        cuda_device=-1,
    )
    return trainer


def read_data(reader: DatasetReader) -> Tuple[List[Instance], List[Instance]]:
    print("Reading data")
    training_data = list(reader.read("../../data/train_human_annotation.txt"))
    validation_data = list(reader.read("../../data/dev_human_annotation.txt"))
    return training_data, validation_data


def run_training_loop():
    dataset_reader = build_data_reader()
    train_data, dev_data = read_data(dataset_reader)
    vocab = build_vocab(train_data + dev_data)
    model = build_model(vocab)

    train_loader, dev_loader = build_data_loaders(train_data, dev_data)
    train_loader.index_with(vocab)
    dev_loader.index_with(vocab)

    # You obviously won't want to create a temporary file for your training
    # results, but for execution in binder for this guide, we need to do this.
    with tempfile.TemporaryDirectory() as serialization_dir:
        trainer = build_trainer(model, serialization_dir, train_loader, dev_loader)
        print("Starting training")
        trainer.train()
        print("Finished training")

    return model, dataset_reader

@Predictor.register("intent_slot_classifier")
class SentenceClassifierPredictor(Predictor):
    def predict(self, sentence: str) -> JsonDict:
        # This method is implemented in the base class.
        return self.predict_json({"sentence": sentence})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        return self._dataset_reader.text_to_instance(sentence)

def BERTtoMT5_data_convert(batch,output_probs):
    inputs = (self.data_list[index].split("<|intent|>")[0].split("<|endoftext|>")[1]).strip()

    labels = (self.data_list[index].split("<|endofcurrentuser|>")[1].split("<|endofintent|>")[
                  0] + ' <|endofintent|>').strip()
    input_ids = [inputs, labels]
    return input_ids

def generate(model, tokenizer, test_list, args, device):
    """
    Use model for information
    :param model:Slected best model
    :param tokenizer:Tokenizer object of pre training model
    :param test_list:Test dataset
    :param args:Experimental parameters
    :param device:CPU or GPU
    :return:NULL
    """
    logger.info('starting generating')
    save_path = open(args.save_path, 'w', encoding='utf-8')
    # test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn1)
    joint_acc = 0
    count = 0
    model.eval()
    dialogue_all = []
    dialogue_dict = {}
    for dialogue in test_list:

        dialogue_dict['' + str(count)] = {
            'target_intent': [],
            'generated_intent': [],
            'target_action': [],
            'generated_action': [],
            'target_response': [],
            'generated_response': []
        }

        # process dialogue
        dialogue_inputs = []
        dialogue_groundtruth = []
        decoder_inputs = []
        outputs = []
        for turns in dialogue.split('\n'):
            if args.task == 'nlu':
                # generate intent
                ### all
                if args.input_type == 'without_context':
                    dialogue_inputs.append(turns.split("<|intent|>")[0].split("<|endofcontext|>")[1])
                else:
                    dialogue_inputs.append(turns.split("<|intent|>")[0].split("<|endoftext|>")[1])
                dialogue_groundtruth.append(turns.split("<|endofcurrentuser|>")[1].split("<|endoftext|>")[0])

            # generate action
            if args.task == 'pl':
                if args.input_type == 'without_context':
                    dialogue_inputs.append(turns.split("<|action|>")[0].split("<|endofcontext|>")[1])
                elif args.input_type == 'without_knowledge':
                    dialogue_inputs.append(turns.split("<|endofintent|>")[0].split("<|endoftext|>")[
                                               1] + ' <|endofintent|>')
                else:
                    dialogue_inputs.append(turns.split('<|action|>')[0].split('<|endoftext|>')[1])
                # decoder_inputs.append(turns.split('<|endofcurrentuser|>')[1].split('<|action|>')[0])
                dialogue_groundtruth.append('<|action|> ' + turns.split('<|action|>')[1].split('<|response|>')[0])
                # dialogue_groundtruth.append(turns.split('<|endofintent|>')[1].split('<|endoftext|>')[0])

            if args.task == 'nlg':
                if args.input_type == 'without_context':
                    dialogue_inputs.append(turns.split("<|response|>")[0].split("<|endofcontext|>")[1])
                elif args.input_type == 'without_knowledge':
                    dialogue_inputs.append(turns.split("<|endofintent|>")[0].split("<|endoftext|>")[
                                               1] + ' <|endofintent|> <|action|>' +
                                           turns.split('<|action|>')[1].split('<|response|>')[0])
                else:
                    # generate response
                    # dialogue_inputs.append(turns.split('<|knowledge|>')[0].split('<|endoftext|>')[1] \
                    #                      +turns.split('<|endofknowledge|>')[1].split('<|response|>')[0])
                    dialogue_inputs.append(turns.split('<|response|>')[0].split('<|endoftext|>')[1])
                dialogue_groundtruth.append(turns.split('<|endofaction|>')[1].split('<|endoftext|>')[0])

        # model generate

        inputs = tokenizer(dialogue_inputs, return_tensors="pt", padding=True, max_length=100).to(device)
        # decoder_inputs = tokenizer(decoder_inputs, return_tensors="pt", padding=True, max_length=100).to(device)
        # print(inputs)
        # outputs = model.generate(inputs["input_ids"], max_length=100, forced_bos_token_id=tokenizer.encode('<en>')[0])
        if args.generate_type == 'end2end':
            # for index in range(len(dialogue_inputs)):
            # inputs = tokenizer(dialogue_inputs, return_tensors="pt").to(device)
            # print(decoder_inputs[index])
            # decoder_input_ids = tokenizer(decoder_inputs[index], return_tensors="pt").to(device)
            outputs = model.generate(inputs["input_ids"], max_length=200)
            # print(outputs)
        else:
            outputs = []
            break_tokens = tokenizer.encode('</s>')

            # print('count:',count)
            ty = 'groundtruth'
            if ty == 'predicted':
                for turns in dialogue.split('\n'):
                    get_intent = False
                    get_action = False
                    inputs = tokenizer(turns.split('<|intent|>')[0].split('<|endofcontext|>')[1],
                                       return_tensors="pt").to(device)
                    knowledge = turns.split('<|endofintent|>')[1].split('<|action|>')[0]
                    # print(knowledge)

                    indexed_tokens = tokenizer.encode('<|intent|>')[:-1]
                    tokens_tensor = torch.tensor(indexed_tokens).to(device).unsqueeze(0)
                    # print(inputs, inputs['input_ids'].size(), tokens_tensor.size())
                    predicted_index = 0
                    predicted_text = ''
                    try:
                        while predicted_index != break_tokens[0]:
                            predictions = model(**inputs, decoder_input_ids=tokens_tensor)[0]
                            predicted_index = torch.argmax(predictions[0, -1, :]).item()
                            # print("pre")
                            # temp = re.sub('[^\u4e00-\u9fa5]','',tokenizer.decode(predicted_index))
                            temp = re.sub('[a-zA-Z<>|]', '', tokenizer.decode(predicted_index))
                            # print("temp:", temp)
                            if temp != '':
                                if not get_intent and temp in predicted_text:
                                    indexed_tokens += [tokenizer.encode('<|endofintent|>')[0]]
                                    # get_intent = True
                                elif get_intent and not get_action and temp in predicted_text.split('<|action|>')[1]:
                                    indexed_tokens += [tokenizer.encode('<|endofaction|>')[0]]
                                    # get_action = True
                                elif get_intent and get_action and temp in predicted_text.split('<|response|>')[1]:
                                    indexed_tokens += [tokenizer.encode('<|endofresponse|>')[0]]
                                else:
                                    indexed_tokens += [predicted_index]
                            # print(predicted_index, tokenizer.decode(predicted_index))
                            else:
                                indexed_tokens += [predicted_index]
                            # print("indexed_tokens:", indexed_tokens)
                            predicted_text = tokenizer.decode(indexed_tokens)
                            # print('predicted_text:',predicted_text)
                            '''temp = longestDupSubstring(re.sub('[^\u4e00-\u9fa5]','',predicted_text))
                            print('temp:',temp)'''
                            '''if temp != '':
                                if '<|endofintent|>' not in predicted_text:
                                    indexed_tokens += [tokenizer.encode('<|endofintent|>')[0]]
                                elif '<|endofaction|>' not in predicted_text:
                                    indexed_tokens += [tokenizer.encode('<|endofaction|>')[0]]
                                elif '<|endofresponse|>' not in predicted_text:
                                    indexed_tokens += [tokenizer.encode('<|endofresponse|>')[0]]'''
                            # print("predicted_text:", predicted_text)
                            if '<|endofintent|>' in predicted_text and not get_intent:
                                # print("predicted_text", predicted_text)
                                get_intent = True
                                # generated_intents.append('<|continue|>'.join(predicted_text.split('<|intent|>')[1].split('<|continue|>')[0:-2]))
                                indexed_tokens = tokenizer.encode(predicted_text + knowledge + '<|action|>')[:-1]

                            if '<|endofaction|>' in predicted_text and not get_action:
                                get_action = True
                                # generated_actions.append(
                                #   '<|continue|>'.join(predicted_text.split('<|action|>')[1].split('<|continue|>')[:-2]))
                                '''indexed_tokens = tokenizer.encode(
                                    predicted_text.split('<|action|>')[0] + '<|action|> {} <|endofaction|> <|response|>'.format(
                                        '<|continue|>'.join(predicted_text.split('<|action|>')[1].split('<|continue|>')[:-2])))[1:-1]'''
                                indexed_tokens = tokenizer.encode(predicted_text + '<|response|>')[:-1]

                            predicted_text = tokenizer.decode(indexed_tokens)
                            tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
                            # print('tokens_tensor:', tokens_tensor.size())
                            if tokenizer.decode(indexed_tokens).endswith('<|endofresponse|>'):
                                break
                            if tokens_tensor.size(-1) > 200:
                                indexed_tokens = tokenizer.encode(predicted_text + ' <|endofresponse|>')
                                break
                    except RuntimeError:
                        pass
                    predicted_text = tokenizer.decode(indexed_tokens)
                    # print(predicted_text)
                    outputs.append(indexed_tokens)

            else:
                for turns in dialogue.split('\n'):
                    get_action = False
                    inputs = tokenizer(turns.split('<|intent|>')[0].split('<|endoftext|>')[1], return_tensors="pt").to(
                        device)
                    # knowledge = turns.split('<|endofintent|>')[1].split('<|action|>')[0]
                    # print(knowledge)

                    indexed_tokens = tokenizer.encode(
                        turns.split('<|endofcurrentuser|>')[1].split('<|action|>')[0] + ' <|action|>')[:-1]
                    response = turns.split('<|endofcurrentuser|>')[1].split('<|response|>')[0]
                    indexed_actions = ''
                    indexed_response = ''

                    tokens_tensor = torch.tensor(indexed_tokens).to(device).unsqueeze(0)
                    # print(inputs, inputs['input_ids'].size(), tokens_tensor.size())
                    predicted_index = 0
                    predicted_text = ''
                    try:
                        while predicted_index != break_tokens[0]:
                            predictions = model(**inputs, decoder_input_ids=tokens_tensor)[0]
                            predicted_index = torch.argmax(predictions[0, -1, :]).item()
                            # print('predicted_text:',predicted_text)
                            # print("pre")
                            # temp = re.sub('[^\u4e00-\u9fa5]','',tokenizer.decode(predicted_index))
                            temp = re.sub('[a-zA-Z<>|]', '', tokenizer.decode(predicted_index))
                            # print("temp:", temp)
                            if temp != '':
                                if not get_action and temp in predicted_text.split('<|action|>')[1]:
                                    indexed_tokens += [tokenizer.encode('<|endofaction|>')[0]]
                                    # get_action = True
                                elif get_action and temp in predicted_text.split('<|response|>')[1]:
                                    indexed_tokens += [tokenizer.encode('<|endofresponse|>')[0]]
                                else:
                                    indexed_tokens += [predicted_index]
                            # print(predicted_index, tokenizer.decode(predicted_index))
                            else:
                                indexed_tokens += [predicted_index]
                            # print("indexed_tokens:", indexed_tokens)
                            predicted_text = tokenizer.decode(indexed_tokens)

                            if '<|endofaction|>' in predicted_text and not get_action:
                                get_action = True
                                indexed_tokens = tokenizer.encode(response + '<|response|>')[:-1]
                                indexed_actions = tokenizer.encode(predicted_text)[:-1]

                            predicted_text = tokenizer.decode(indexed_tokens)
                            tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
                            # print('tokens_tensor:', tokens_tensor.size())
                            if tokenizer.decode(indexed_tokens).endswith('<|endofresponse|>'):
                                indexed_response = tokenizer.encode(predicted_text.split('<|endofknowledge|>')[1])
                                break
                            if tokens_tensor.size(-1) > 300:
                                indexed_response = tokenizer.encode(
                                    predicted_text.split('<|endofknowledge|>')[1] + ' <|endofresponse|>')
                                break
                    except RuntimeError:
                        pass
                    # predicted_text = tokenizer.decode(indexed_tokens)
                    # print(predicted_text)
                    if len(indexed_actions) == 0:
                        indexed_actions = tokenizer.encode('<|action|> <|endofaction|>')[:-1]
                    outputs.append(indexed_actions + indexed_response)
        # tokenizer decode and
        for index in range(len(outputs)):
            # print(len(outputs))
            # print(tokenizer.decode(outputs[index]))
            generation = re.sub('</s>', '', re.sub('<pad>', '', tokenizer.decode(outputs[index])))
            # print("generation", generation)
            # generation = tokenizer.decode(outputs[index]).split('</s>')[0].split('<pad>')[1]
            # print("groundtruth:", dialogue_groundtruth[index])
            if args.task == 'nlu':
                dialogue_dict['' + str(count)]['target_intent'].append(
                    dialogue_groundtruth[index].split('<|intent|>')[1].split('<|endofintent|>')[0])
            elif args.task == 'pl':
                dialogue_dict['' + str(count)]['target_action'].append(
                    dialogue_groundtruth[index].split('<|action|>')[1].split('<|endofaction|>')[0])
            else:
                dialogue_dict['' + str(count)]['target_response'].append(
                    dialogue_groundtruth[index].split('<|response|>')[1].split('<|endofresponse|>')[0])
            if '<|intent|>' in generation and '<|endofintent|>' in generation:
                dialogue_dict['' + str(count)]['generated_intent'].append(
                    generation.split('<|intent|>')[1].split('<|endofintent|>')[0])
            else:
                dialogue_dict['' + str(count)]['generated_intent'].append(' ')
            if '<|action|>' in generation and '<|endofaction|>' in generation:
                dialogue_dict['' + str(count)]['generated_action'].append(
                    generation.split('<|action|>')[1].split('<|endofaction|>')[0])
            else:
                dialogue_dict['' + str(count)]['generated_action'].append(' ')
            if '<|response|>' in generation and '<|endofresponse|>' in generation:
                dialogue_dict['' + str(count)]['generated_response'].append(
                    generation.split('<|response|>')[1].split('<|endofresponse|>')[0])
            else:
                dialogue_dict['' + str(count)]['generated_response'].append(' ')
        print("count:", count)
        count += 1
    json.dump(dialogue_dict, save_path, indent=1, ensure_ascii=False)
    save_path.close()

if __name__ == "__main__":
    dataset_reader = build_data_reader()
    model = load_archive("model.tar.gz").model
    vocab = model.vocab
    test_data = list(dataset_reader.read("../../data/test_human_annotation.txt"))
    data_loader = SimpleDataLoader(test_data, batch_size=10)
    data_loader.index_with(model.vocab)
    results = evaluate(model, data_loader)
    print(results)

    cuda_device = 0
    iterator = iter(data_loader)
    generator_tqdm = Tqdm.tqdm(iterator)

    # Number of batches in instances.
    batch_count = 0
    # Number of batches where the model produces a loss.
    loss_count = 0
    # Cumulative weighted loss
    total_loss = 0.0
    # Cumulative weight across all batches.
    total_weight = 0.0

    for batch in generator_tqdm:
        batch_count += 1
        batch = util.move_to_device(batch, cuda_device)
        output_dict = model(**batch)
        loss = output_dict.get("loss")
        probs = output_dict.get("probs")


    checkpoint = ""
    model = modeling_mt5_cl.MT5ForConditionalGeneration.from_pretrained(checkpoint)
    generate(model, tokenizer, test_list1, args, device)