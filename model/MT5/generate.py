"""
    Use model for information
    :param model:Slected best model
    :param tokenizer:Tokenizer object of pre training model
    :param test_list:Test dataset
    :param args:Experimental parameters
    :param device:CPU or GPU
    :return:NULL
    """
import argparse
import json
import re
import torch
import modeling_mt5_cl
from transformers import MT5Tokenizer


def get_tokenizer():
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['<|user|>', '<|system|>', '<|intent|>', '<|endofintent|>',
                                       '<|action|>', '<|endofaction|>', '<|response|>', '<|endofresponse|>'
            , '<|knowledge|>', '<|endofknowledge|>', '<|continue|>', '<|k|>']})
    return tokenizer

def load_model():
    device = 0
    nlu_checkpoint = "nlu_model_epoch30"
    pl_checkpoint = "pl_model_epoch19"
    nlg_checkpoint = "nlg_model_epoch30"

    model_nlu = modeling_mt5_cl.MT5ForConditionalGeneration.from_pretrained(nlu_checkpoint)
    model_pl = modeling_mt5_cl.MT5ForConditionalGeneration.from_pretrained(pl_checkpoint)
    model_nlg = modeling_mt5_cl.MT5ForConditionalGeneration.from_pretrained(nlg_checkpoint)
    model_nlu.to(device)
    model_pl.to(device)
    model_nlg.to(device)
    model_nlu.eval()
    model_pl.eval()
    model_nlg.eval()
    return model_nlu,model_pl,model_nlg


'''
    Parameters:
    history_dialog: list=[{"user":"str"},{"system":"str"},{"user":"str"},{"system":"str"},……] 
    current_user: 'str'
    
    Return： {"response":"str"}
'''
def get_response(history_dialog,current_user):

    # 载入model 和 tokenizer，为了提高速度，可以把这两句代码放到全局去
    model_nlu, model_pl, model_nlg = load_model()
    tokenizer = get_tokenizer()

    device = 0
    context = ''
    knowledge = ''

    # process dialogue
    dialogue_inputs = []

    # construct model inputs
    if current_user == '' and not history_dialog:
        context = ''
    else:
        for turn in history_dialog:
            if turn.__contains__("user"):
                context += "<|user|> " + turn["user"]
            if turn.__contains__("system"):
                context += " <|system|>" + turn["system"]+" "
    turns = "<|context|> "+context+" <|endofcontext|> " + " <|currentuser|> "+current_user+" <|endofcurrentuser|>"
    dialogue_inputs.append(turns)

    # model generate
    inputs = tokenizer(dialogue_inputs, return_tensors="pt", padding=True, max_length=100).to(device)
    outputs = model_nlu.generate(inputs["input_ids"], max_length=200)
    # print(outputs)
    intent = ''
    for index in range(len(outputs)):
        generation = re.sub('</s>', '', re.sub('<pad>', '', tokenizer.decode(outputs[index])))
        if '<|intent|>' in generation and '<|endofintent|>' in generation:
            intent = generation.split('<|intent|>')[1].split('<|endofintent|>')[0]
    print("intent:",intent)

    dialogue_inputs[0] += " <|intent|> " + intent + "<|endofintent|>"
    dialogue_inputs[0] += " <|knowledge|>" + knowledge + "<|endofknowledge|>"
    # print(dialogue_inputs)
    inputs = tokenizer(dialogue_inputs, return_tensors="pt", padding=True, max_length=100).to(device)
    outputs = model_pl.generate(inputs["input_ids"], max_length=200)
    action = ''
    for index in range(len(outputs)):
        generation = re.sub('</s>', '', re.sub('<pad>', '', tokenizer.decode(outputs[index])))
        if '<|action|>' in generation and '<|endofaction|>' in generation:
            action = generation.split('<|action|>')[1].split('<|endofaction|>')[0]
    print("action:",action)

    dialogue_inputs[0] += " <|action|> " + action + " <|endofaction|>"

    inputs = tokenizer(dialogue_inputs, return_tensors="pt", padding=True, max_length=100).to(device)
    outputs = model_nlg.generate(inputs["input_ids"], max_length=200)
    response = ''
    for index in range(len(outputs)):
        generation = re.sub('</s>', '', re.sub('<pad>', '', tokenizer.decode(outputs[index])))
        if '<|response|>' in generation and '<|endofresponse|>' in generation:
            response = generation.split('<|response|>')[1].split('<|endofresponse|>')[0]
    return {"response":response}



def setup_train_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行训练')
    parser.add_argument('--vocab_path', default='vocabulary/vocab_small.txt', type=str, required=False,
                        help='选择词库,UNUSED IN THIS CODE')
    parser.add_argument('--pretrained_model', default='nlg_model_epoch30', type=str, required=False, help='预训练的MT5模型的路径')
    parser.add_argument("--ft2", action='store_true', help='second fine tune')
    parser.add_argument('--generate_type', default='end2end', type=str, required=False, help='generate end2end ')
    parser.add_argument('--model', default='train', type=str, required=False, help='train or test ')
    parser.add_argument('--tokenizer_path', default='tokenizer', type=str, required=False, help='tokenizer path')
    parser.add_argument('--task', default='nlg', type=str, required=False, help='task: nlu,pl,nlg')
    parser.add_argument('--evaluate_type', default='acc', type=str, required=False, help='task: nlu,pl,nlg')
    parser.add_argument('--input_type', default='', type=str, required=False,
                        help='ablation experiment type: WOC,WOK,all')
    parser.add_argument('--cl', action='store_true', help='Add contrastive learning')
    return parser.parse_args()
def run():
    args = setup_train_args()
    joint_acc = 0
    count = 0
    device = 0

    nlu_checkpoint = "nlu_model_epoch30"
    pl_checkpoint = "pl_model_epoch19"
    nlg_checkpoint = "nlg_model_epoch30"

    model_nlu = modeling_mt5_cl.MT5ForConditionalGeneration.from_pretrained(nlu_checkpoint)
    model_pl = modeling_mt5_cl.MT5ForConditionalGeneration.from_pretrained(pl_checkpoint)
    model_nlg = modeling_mt5_cl.MT5ForConditionalGeneration.from_pretrained(nlg_checkpoint)
    model_nlu.to(device)
    model_pl.to(device)
    model_nlg.to(device)
    model_nlu.eval()
    model_pl.eval()
    model_nlg.eval()

    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['<|user|>', '<|system|>', '<|intent|>', '<|endofintent|>',
                                       '<|action|>', '<|endofaction|>', '<|response|>', '<|endofresponse|>'
            , '<|knowledge|>', '<|endofknowledge|>', '<|continue|>', '<|k|>']})


    # user_f = open("user_utterance.txt", "r", encoding='utf-8')

    current_user = ''
    context = ''
    response = ''
    knowledge = ''
    context_list = []
    dialogue_all = []
    dialogue_dict = {}

    while True:
        dialogue_dict['' + str(count)] = {
            'generated_intent': [],
            'generated_action': [],
            'generated_response': []
        }

        # process dialogue
        dialogue_inputs = []
        decoder_inputs = []
        outputs = []

        # get dialog inputs
        if current_user == '' and response == '':
            context = ''
        else: context += "<|user|> " + current_user + " <|system|>" + response
        # current_user = user_f.readline().split('\n')[0]
        current_user=input("current_user:")
        # if user end，exit
        if current_user == '': break
        turns = "<|context|> "+context+" <|endofcontext|> " + " <|currentuser|> "+current_user+" <|endofcurrentuser|>"
        dialogue_inputs.append(turns)

        # model generate
        if args.generate_type == 'end2end':
            # print(dialogue_inputs)
            inputs = tokenizer(dialogue_inputs, return_tensors="pt", padding=True, max_length=100).to(device)
            outputs = model_nlu.generate(inputs["input_ids"], max_length=200)
            # print(outputs)
            intent = ''
            for index in range(len(outputs)):
                generation = re.sub('</s>', '', re.sub('<pad>', '', tokenizer.decode(outputs[index])))
                if '<|intent|>' in generation and '<|endofintent|>' in generation:
                    intent = generation.split('<|intent|>')[1].split('<|endofintent|>')[0]
                    dialogue_dict['' + str(count)]['generated_intent'].append(intent)
                else:
                    dialogue_dict['' + str(count)]['generated_intent'].append(' ')
            print("intent:",intent)

            dialogue_inputs[0] += " <|intent|> " + intent + "<|endofintent|>"
            dialogue_inputs[0] += " <|knowledge|>" + knowledge + "<|endofknowledge|>"
            # print(dialogue_inputs)
            inputs = tokenizer(dialogue_inputs, return_tensors="pt", padding=True, max_length=100).to(device)
            outputs = model_pl.generate(inputs["input_ids"], max_length=200)
            action = ''
            for index in range(len(outputs)):
                generation = re.sub('</s>', '', re.sub('<pad>', '', tokenizer.decode(outputs[index])))
                if '<|action|>' in generation and '<|endofaction|>' in generation:
                    action = generation.split('<|action|>')[1].split('<|endofaction|>')[0]
                    dialogue_dict['' + str(count)]['generated_action'].append(action)
                else:
                    dialogue_dict['' + str(count)]['generated_action'].append(' ')
            print("action:",action)

            dialogue_inputs[0] += " <|action|> " + action + " <|endofaction|>"

            inputs = tokenizer(dialogue_inputs, return_tensors="pt", padding=True, max_length=100).to(device)
            outputs = model_nlg.generate(inputs["input_ids"], max_length=200)
            response = ''
            for index in range(len(outputs)):
                generation = re.sub('</s>', '', re.sub('<pad>', '', tokenizer.decode(outputs[index])))
                if '<|response|>' in generation and '<|endofresponse|>' in generation:
                    response = generation.split('<|response|>')[1].split('<|endofresponse|>')[0]
                    dialogue_dict['' + str(count)]['generated_response'].append(response)
                else:
                    dialogue_dict['' + str(count)]['generated_response'].append(' ')
            print("response:",response)

        else:
            outputs = []
            break_tokens = tokenizer.encode('</s>')

            # print('count:',count)
            ty = 'predicted'
            if ty == 'predicted':
                get_intent = False
                get_action = False
                inputs = tokenizer(turns.split('<|endofcontext|>')[1],return_tensors="pt").to(device)
                knowledge = knowledge
                # print(knowledge)

                indexed_tokens = tokenizer.encode('<|intent|>')[:-1]
                tokens_tensor = torch.tensor(indexed_tokens).to(device).unsqueeze(0)
                predicted_index = 0
                predicted_text = ''
                try:
                    while predicted_index != break_tokens[0]:
                        predictions = model(**inputs, decoder_input_ids=tokens_tensor)[0]
                        predicted_index = torch.argmax(predictions[0, -1, :]).item()
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

                        if '<|endofintent|>' in predicted_text and not get_intent:
                            # print("predicted_text", predicted_text)
                            get_intent = True
                            indexed_tokens = tokenizer.encode(predicted_text + knowledge + '<|action|>')[:-1]

                        if '<|endofaction|>' in predicted_text and not get_action:
                            get_action = True
                            indexed_tokens = tokenizer.encode(predicted_text + '<|response|>')[:-1]

                        predicted_text = tokenizer.decode(indexed_tokens)
                        tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
                        # print('tokens_tensor:', tokens_tensor.size())
                        if tokenizer.decode(indexed_tokens).endswith('<|endofresponse|>'):
                            break
                        if tokens_tensor.size(-1) > 200:
                            indexed_tokens = tokenizer.encode(predicted_text + ' <|endofresponse|>')
                            print("stop generating as too long")
                            break
                except RuntimeError:
                    pass
                predicted_text = tokenizer.decode(indexed_tokens)
                # print(predicted_text)
                outputs.append(indexed_tokens)

            else:
                for turns in dialogue_inputs:
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
                    if len(indexed_actions) == 0:
                        indexed_actions = tokenizer.encode('<|action|> <|endofaction|>')[:-1]
                    outputs.append(indexed_actions + indexed_response)

if __name__ == "__main__":
    history_dialog = [{"user": "str"}, {"system": "str"}, {"user": "str"}, {"system": "str"}]
    current_user = ""
    get_response(history_dialog,current_user)