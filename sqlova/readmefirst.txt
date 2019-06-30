train, test, val 中的 "*_tok.json", "*.tables.json", "*.db" 全部在 "sqlova/wikisql/data/tianchi" 文件夹下

pip install pytorch_pretrained_bert

运行命令：(指定参数--user, 罗凯0，晋豪1，刘超2)

python train.py --seed 1 --bS 4 --accumulate_gradients 2 --bert_type_abb zhS --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_leng 222 --user 1
