train, test, val �� "*_tok.json", "*.tables.json", "*.db" ������ "sqlova/wikisql/data/tianchi" �ļ�����

�޸�train.py 691�е�path_hΪ����"sqlova/wikisql"�ľ���·��

pip install pytorch_pretrained_bert

���д������ӣ�

python train.py --seed 1 --bS 4 --accumulate_gradients 2 --bert_type_abb zhS --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_leng 222
