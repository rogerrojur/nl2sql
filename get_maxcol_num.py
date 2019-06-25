import json


def count_lines(fname):
    with open(fname) as f:
        return sum(1 for line in f)


def get_error_count(fname):
    count = 0
    with open(fname) as fs:
        for line in fs:
            if 'tok_error' in line:
                count += 1

    return count / count_lines(fname)
            

if __name__ == '__main__':
    train_part = get_error_count('./train_tok.json')
    print('train_tok.json中tok_error的数据：%.3f.' % train_part)
    val_part = get_error_count('./val_tok.json')
    print('val_tok.json中tok_error的数据：%.3f.' % val_part)