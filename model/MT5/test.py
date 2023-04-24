import re,sys
import torch

def count_test():
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

    #gpt style
    datas = open("../../data/test_human_annotation.txt", 'r', encoding='utf-8').read().split('\n\n')
    num_data = len(datas)
    generated_dict = {}
    count = 0
    inside_count = 0

    for data_id, data in enumerate(datas[0:-1]):
        # print('\n\n\n[{}/{}] \r'.format(data_id, num_data), end='')
        sys.stdout.flush()
        count += 1
        for turn_id, turn in enumerate(data.split('\n')):
            inside_count += 1
    print(count,inside_count) #800,6597
    exit(0)

    # mt5 style
    test1_temp = open("../../data/test_human_annotation.txt", "r", encoding='utf-8').read().split('\n\n')
    test_list1 = []
    count=0
    inside_count = 0
    for data in test1_temp[0:-1]:
        test_list1 += data.split('\n')
    print(test_list1[0])
    for dialogue in test_list1:
        for turns in dialogue.split('\n'):
            inside_count += 1
        count+=1
    print(count) #6597 = count = inside_count


    # bert style
    count=0
    not_null_count = 0
    with open("../../data/test_human_annotation.txt", 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
            start_index = 0
            end_index = -1
            # add Separator
            Separator = 'intent'
            for line in lines[start_index:end_index]:
                if line != '':
                    text = line.strip().split('<|' + Separator + '|>')[0]
                    intent = re.sub('[\u4e00-\u9fa5]', '',
                                    line.strip().split('<|endof' + Separator + '|>')[0].split('<|' + Separator + '|>')[
                                        1])
                    intent_slot = [x.strip() for x in intent.split('<|continue|>') if x.strip() in intent_slot1.keys()]
                    count += 1
                    if len(intent_slot) != 0:
                        not_null_count += 1
    print(count)
    print(not_null_count)

f = torch.tensor([0.5,0.2,0.3,0.9])
print("f.type():",f.type())
#改变数据类型
print(f.int())
print(f.type(torch.IntTensor))
#取整
print(torch.round(f))
