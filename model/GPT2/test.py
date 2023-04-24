def preprocess_raw_data(raw_path, tokenized_path, tokenizer, n_ctx):
    """
    Process the original corpus and convert the original corpus into token ID for train
    :param raw_path: Original corpus
    :param tokenizer_path:Token id file path corresponding to the original corpus
    :param tokenizer:Tokenizer object of pre training model
    :param n_ctx:Maxmium truncation length
    :return:
    """
    logger.info("tokenizing raw data,raw data path:{}, token output path:{}".format(raw_path,
                                                                                    tokenized_path))
    special_token = tokenizer.encode(
        '<|intent|> <|endofintent|> <|action|> <|endofaction|> <|response|> <|endofresponse|>')[1:-1]
    special_token[0] = tokenizer.encode('<|intent|>')[1]
    special_token[1] = tokenizer.encode('<|endofintent|>')[1]
    special_token[2] = tokenizer.encode('<|action|>')[1]
    special_token[3] = tokenizer.encode('<|endofaction|>')[1]
    special_token[4] = tokenizer.encode('<|response|>')[1]
    special_token[5] = tokenizer.encode('<|endofresponse|>')[1]
    user_id = tokenizer.encode('<|user|>')[1]
    currentuser_id = tokenizer.encode('<|currentuser|>')[1]
    miss_count = 0
    with open(raw_path, 'rb') as f:
        data = f.read().decode("utf-8")
    if "\r\n" in data:
        train_data = data.split("\r\n\r\n")
    else:
        train_data = data.split("\n\n")
    logger.info("there are {} dialogue in raw dataset".format(len(train_data)))
    with open(tokenized_path, "w", encoding="utf-8") as f:
        for dialogue_index, dialogue in enumerate(tqdm(train_data)):
            if "\r\n" in data:
                utterances = dialogue.split("\r\n")
            else:
                utterances = dialogue.split("\n")

            for utterance in utterances:
                try :
                    if utterance !='':
                        utterance = '<|endoftext|> '+ utterance.split('<|endofcontext|>')[1]
                    special_index = [0] * 6
                    dialogue_ids = tokenizer.encode(utterance)[1:-1]


                    user_count = 0
                    user_index = 0
                    if n_ctx-6 < len(dialogue_ids):#
                        for index, id in enumerate(dialogue_ids):
                            if id == currentuser_id:
                                currentuser_index = index
                            if id == user_id and user_count == 2:
                                user_index = index
                                #dialogue_ids = dialogue_ids[0:index]+dialogue_ids[-(n_ctx-6-index):]
                            if id == user_id and user_count < 2:
                                user_count += 1

                        if len(dialogue_ids[currentuser_index:]) > n_ctx-6:
                            miss_count += 1
                            continue
                        elif len(dialogue_ids[0:user_index])+len(dialogue_ids[currentuser_index:]) > n_ctx-6 :
                             dialogue_ids = tokenizer.encode('<|endoftext|> <|context|> <|endofcontext|>')+ dialogue_ids[currentuser_index:]
                        else:
                            dialogue_ids = dialogue_ids[0:user_index] + dialogue_ids[-(n_ctx - 6 - index):]
                    # print(len(dialogue_ids))
                    # dialogue_ids.extend([tokenizer.convert_tokens_to_ids(word) for word in utterance])
                    # dialogue_ids.extend(tokenizer.encode(utterance))
                    # dialogue_ids.append(tokenizer.sep_token_id)
                    # print(len(dialogue_ids))
                    #dialogue_ids = dialogue_ids[-n_ctx:]
                    # Perform subscript search for three special flags
                    for index, id in enumerate(dialogue_ids):
                        if id == special_token[0]:
                            special_index[0] = index
                        if id == special_token[1]:
                            special_index[1] = index
                        if id == special_token[2]:
                            special_index[2] = index
                        if id == special_token[3]:
                            special_index[3] = index
                            special_index[4] = index + 1
                    special_index[5] = len(dialogue_ids) - 2
                    if special_index[0] == 0:
                        continue
                    special_index.extend(dialogue_ids)  # Add the subscripts of 6 special tokens to the front

                    for id in special_index:
                        f.write(str(id) + ' ')
                    #The last record does not add a newline character
                    if dialogue_index < len(train_data) - 1:
                        f.write("\n")
                except IndexError:
                    pass
        print("miss_count:", miss_count)
    logger.info("finish preprocessing raw data,the result is stored in {}".format(tokenized_path))