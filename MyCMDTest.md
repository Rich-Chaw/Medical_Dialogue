conda activate dialog
cd Medical-Dialogue-main/model/
# bert
cd model/BERT
allennlp train bert_wwm_nlu.json --include-package nlu -s BERT-WWM-NLU
allennlp train bert_med_nlu.json --include-package nlu -s BERT-BASE-CHINESE3-NEW-NLU
allennlp train bert_wwm_nlu.json --include-package nlu -s BERT-WWM2-NLU
allennlp train bert_bilstm_crf_nlu.json --include-package nlu_lstm_crf -s BERT-BILSTM-CRF-NLU

# GPT2
cd model/GPT2
# 含raw，对数据进行了处理
python -m torch.distributed.launch --nproc_per_node=1 --master_port 22665 train_gpt2.py --device 0 --epochs 30 --raw --batch_size 4 --log_step 100 --eval_all_checkpoints --gradient_accumulation 4 --num_workers 4 --log_path ../../common/log/gpt2_test.log --writer_dir ../../common/tensorboard_summary/tensorboard_gpt2_test --dialogue_model_output_path ../../common/model/gpt2_test_model/
# 不含raw的普通版
python -m torch.distributed.launch --nproc_per_node=1 --master_port 22665 train_gpt2.py --device 0 --epochs 30 --batch_size 4 --log_step 100 --eval_all_checkpoints --gradient_accumulation 4 --num_workers 4 --log_path ../../common/log/gpt2_test.log --writer_dir ../../common/tensorboard_summary/tensorboard_gpt2_test --dialogue_model_output_path ../../common/model/gpt2_test_model/ --inference_result ../../common/output/gpt2_test
# ft2
python -m torch.distributed.launch --nproc_per_node=1 --master_port 22665 train_gpt2.py --device 0 --epochs 30 --batch_size 4 --log_step 100 --eval_all_checkpoints --gradient_accumulation 4 --num_workers 4 --log_path ../../common/log/gpt2_test_ft2.log --writer_dir ../../common/tensorboard_summary/tensorboard_gpt2_test_ft2 --dialogue_model_output_path ../../common/model/gpt2_test_ft2_model/ --ft2 --inference_result ../../common/output/gpt2_test_ft2
# ft2 based on checkpoint
nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port 22665 train_gpt2.py --device 0 --epochs 30 --batch_size 4 --log_step 100 --eval_all_checkpoints --gradient_accumulation 4 --num_workers 4 --log_path ../../common/log/gpt2_test_ft2_ck.log --writer_dir ../../common/tensorboard_summary/tensorboard_gpt2_test_ft2_ck --dialogue_model_output_path ../../common/model/gpt2_test_ft2_ck_model/ --ft2 --inference_result ../../common/output/gpt2_test_ft2_ck --pretrained_model ../../common/model/gpt2_test_model/model_epoch29 --model_config ../../common/model/gpt2_test_model/model_epoch29/config.json &
# nohup版本
nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port 22665 train_gpt2.py --device 0 --epochs 30 --batch_size 4 --log_step 100 --eval_all_checkpoints --gradient_accumulation 4 --num_workers 4 --log_path ../../common/log/gpt2_test.log --writer_dir ../../common/tensorboard_summary/tensorboard_gpt2_test --dialogue_model_output_path ../../common/model/gpt2_test_model/ --inference_result ../../common/output/gpt2_test > train_gpt2_out.log 2>&1 &
# 便于观察过程的temp版
nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port 22665 train_gpt2.py --device 0 --epochs 30 --batch_size 4 --log_step 100 --eval_all_checkpoints --gradient_accumulation 4 --num_workers 4 --log_path ../../common/temp/gpt2_test.log --writer_dir ../../common/temp/tensorboard_gpt2_test --dialogue_model_output_path ../../common/temp/ --inference_result ../../common/temp/gpt2_test > train_gpt2_out_simple.log 2>&1 &
# evaluation
cd evaluate/
python eva.py --evaluation_path ../common/output/mt5_test_ft2.json --evaluation_task=nlu




# MT5
cd model/MT5


nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port 22665 train_mt5.py --epochs 40 --log_step 100 --pretrained_model google/mt5-small --dialogue_model_output_path ../../common/model/mt5_test_model/ --eval_all_checkpoints --log_path ../../common/log/mt5_test.log --save_path ../../common/output/mt5_test.json --writer_dir ../../common/tensorboard_summary/mt5_test/ --generate_type 'end2end' --task 'nlu' & 

nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port 22665 train_mt5.py --epochs 20 --log_step 100 --pretrained_model google/mt5-small --dialogue_model_output_path ../../common/model/mt5_test_pl_model/ --eval_all_checkpoints --log_path ../../common/log/mt5_test_pl.log --save_path ../../common/output/mt5_test_pl.json --writer_dir ../../common/tensorboard_summary/mt5_test_pl/ --generate_type 'end2end' --task 'nlu' > train_mt2_pl.log 2>&1 &

# add ft2
python -m torch.distributed.launch --nproc_per_node=1 --master_port 22665 train_mt5.py --epochs 30 --log_step 100 --ft2  --dialogue_model_output_path ../../common/model/mt5_test_ft2_model/ --eval_all_checkpoints --log_path ../../common/log/mt5_test_ft2.log --save_path ../../common/output/mt5_test_ft2.json --writer_dir ../../common/tensorboard_summary/mt5_test_ft2/ --generate_type 'end2end' --task 'nlu'

python -m torch.distributed.launch --nproc_per_node=1 --master_port 22665 train_mt5.py --epochs 30 --log_step 100 --ft2  --dialogue_model_output_path ../../common/model/mt5_pl_test_ft2_model/ --eval_all_checkpoints --log_path ../../common/log/mt5_pl_test_ft2.log --save_path ../../common/output/mt5_pl_test_ft2.json --writer_dir ../../common/tensorboard_summary/mt5_pl_test_ft2/ --generate_type 'end2end' --task 'pl'


nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port 22665 train_mt5.py --epochs 30 --log_step 100 --ft2  --dialogue_model_output_path ../../common/model/mt5_nlg_test_ft2_model/ --eval_all_checkpoints --log_path ../../common/log/mt5_nlg_test_ft2.log --save_path ../../common/output/mt5_nlg_test_ft2.json --writer_dir ../../common/tensorboard_summary/mt5_nlg_test_ft2/ --generate_type 'end2end' --task 'nlg' &

nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port 22667 train_mt5.py --epochs 30 --log_step 100 --ft2  --dialogue_model_output_path ../../common/model/mt5_test_ft2_model_0330/ --eval_all_checkpoints --log_path ../../common/log/mt5_test_ft2_0330.log --save_path ../../common/output/mt5_test2_ft_0330.json --writer_dir ../../common/tensorboard_summary/mt5_test2_ft_0330/ --generate_type 'end2end' --task 'nlu' > train_mt2_ft2.log 2>&1 &


# ft2 based on checkpoint
nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port 22665 train_mt5.py --epochs 30 --log_step 100 --ft2  --dialogue_model_output_path ../../common/model/mt5_test_ft2_ck_model/ --eval_all_checkpoints --log_path ../../common/log/mt5_test_ck_ft2.log --save_path ../../common/output/mt5_test2_ck_ft.json --writer_dir ../../common/tensorboard_summary/mt5_test2_ck_ft/ --generate_type 'end2end' --pretrained_model ../../common/model/mt5_test_model/model_epoch30 --model_config ../../common/model/mt5_test_model/model_epoch30/config.json --task 'nlu' &

# test
nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port 22665 train_mt5.py --epochs 30 --log_step 100 --ft2 --model test --eval_all_checkpoints --log_path ../../common/log/mt5_test_ft2_ck_epoch22.log --save_path ../../common/output/mt5_test2_ck_ft_epoch22.json  --generate_type 'end2end' --pretrained_model ../../common/model/mt5_test_ft2_ck_model/model_epoch22 --model_config ../../common/model/mt5_test_ft2_ck_model/model_epoch22/config.json --task 'nlu' &

nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port 22665 train_mt5.py --epochs 30 --log_step 100 --ft2  --dialogue_model_output_path ../../common/model/mt5_pl_test_ft2_model/ --eval_all_checkpoints --log_path ../../common/log/mt5_pl_test_ft2.log --save_path ../../common/output/mt5_pl_test_ft2_epoch19.json --writer_dir ../../common/tensorboard_summary/mt5_pl_test_ft2/ --generate_type 'end2end' --pretrained_model ../../common/model/mt5_pl_test_ft2_model/model_epoch19 --model test --task 'pl' &

# add cl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 22665 train_mt5.py --epochs 30 --log_step 100 --cl --pretrained_model google/mt5-small --ft2  --dialogue_model_output_path ../../common/model/mt5_test_ft2_cl_model/ --eval_all_checkpoints --log_path ../../common/log/mt5_test_ft2_cl.log --save_path ../../common/output/mt5_test_ft2_cl.json --writer_dir ../../common/tensorboard_summary/mt5_test_ft2_cl/ --generate_type 'end2end' --task 'nlu'



# evaluation
python eva.py --evaluation_path ../common/output/gpt2_test_ft2_ck.json --evaluation_task=nlu

python eva.py --evaluation_path ../model/BERT/pretrained_union.json --evaluation_task=nlu

# union
nohup python train_BERT.py > generate_MT5_out.log 2>&1 &
nohup python train_BERT.py > generate_BERT_GPT2_out.log 2>&1 &