import os

from sqlnet.dbengine import DBEngine
import json
import time

path_db = './wikisql/data/tianchi/'
dset_name = 'val'

# The engine for seaching results
engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))

# Return the results queried
query = lambda sql_tmp: engine.execute(sql_tmp['table_id'], sql_tmp['sql']['sel'], sql_tmp['sql']['agg'], sql_tmp['sql']['conds'], sql_tmp['sql']['cond_conn_op'])

fname = os.path.join(path_db, f"{dset_name}.json")
with open(fname, encoding='utf-8') as fs:
    total_count = 0
    start = time.time()
    for line in fs:
        record = json.loads(line)
        break
    while total_count < 10:
        res = query(record)
        total_count += 1
    print('%d times of invoking cost %.2fs.' % (total_count, time.time() - start))
    # print('The number of empty results is %d, the number of result is 0 %d' % (count, tmp))