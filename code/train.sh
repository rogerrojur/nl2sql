# --trained表示使用生成的模型，需要使用此时的生成文件
python train.py --seed 4 --bS 4 --accumulate_gradients 2 --bert_type_abb zhS --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_leng 400 --mode train --token 0 --tepoch 1 --EG
python train.py --seed 4 --bS 4 --accumulate_gradients 2 --bert_type_abb zhS --fine_tune --lr 0.00001 --lr_bert 0.00001 --max_seq_leng 400 --mode train --token 0 --tepoch 1 --EG --trained
python train.py --seed 4 --bS 4 --accumulate_gradients 2 --bert_type_abb zhS --fine_tune --lr 0.000001 --lr_bert 0.000001 --max_seq_leng 400 --mode train --token 0 --tepoch 1 --EG --trained
python train.py --seed 4 --bS 4 --accumulate_gradients 2 --bert_type_abb zhS --fine_tune --lr 0.00000001 --lr_bert 0.00000001 --max_seq_leng 400 --mode train --token 0 --tepoch 1 --EG --trained
python train.py --seed 4 --bS 4 --accumulate_gradients 2 --bert_type_abb zhS --fine_tune --lr 0.000000001 --lr_bert 0.000000001 --max_seq_leng 400 --mode train --token 0 --tepoch 1 --EG --trained