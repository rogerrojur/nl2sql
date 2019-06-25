import json


def count_lines(fname):
    with open(fname) as f:
        return sum(1 for line in f)


def get_maxcol_num(fname):
    max_wc, max_sc = 0, 0
    with open(fname) as fs:
        for line in fs:
            d = json.loads(line)
            sql = d['sql']
            max_sc = max(len(sql['sel']), max_sc)
            max_wc = max(len(sql['conds']), max_wc)

    return max_wc, max_sc
            

if __name__ == '__main__':
    train_wc, train_sc = get_maxcol_num('./data/train/train.json')
    print('train.json中where col的最大个数是%d, sel col的最大个数是%d.' % (train_wc, train_sc))
    val_wc, val_sc = get_maxcol_num('./data/val/val.json')
    print('val.json中where col的最大个数是%d, sel col的最大个数是%d.' % (val_wc, val_sc))