train, test, val 的 "*_tok.json", "*.tables.json", "*.db" 都放在 "sqlova/wikisql/data/tianchi" 文件夹下

修改train.py 691行的path_h为到达"sqlova/wikisql"的绝对路径

pip install pytorch_pretrained_bert

运行代码例子：

python train.py --seed 1 --bS 4 --accumulate_gradients 2 --bert_type_abb zhS --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_leng 222
