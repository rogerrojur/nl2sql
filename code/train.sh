# ~ 0.679, 0.746 训练一个或两个epoch --tepch 2
python train.py --seed 4 --bS 4 --accumulate_gradients 2 --bert_type_abb zhS --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_leng 400 --mode train --token 0 --tepoch 1 --EG
python train.py --seed 4 --bS 4 --accumulate_gradients 2 --bert_type_abb zhS --fine_tune --lr 0.00001 --lr_bert 0.00001 --max_seq_leng 400 --mode train --token 0 --tepoch 1 --EG --trained
python train.py --seed 4 --bS 4 --accumulate_gradients 2 --bert_type_abb zhS --fine_tune --lr 0.000001 --lr_bert 0.000001 --max_seq_leng 400 --mode train --token 0 --tepoch 1 --EG --trained
python train.py --seed 4 --bS 4 --accumulate_gradients 2 --bert_type_abb zhS --fine_tune --lr 0.00000001 --lr_bert 0.00000001 --max_seq_leng 400 --mode train --token 0 --tepoch 1 --EG --trained
python train.py --seed 4 --bS 4 --accumulate_gradients 2 --bert_type_abb zhS --fine_tune --lr 0.000000001 --lr_bert 0.000000001 --max_seq_leng 400 --mode train --token 0 --tepoch 1 --EG --trained