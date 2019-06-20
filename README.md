# nl2sql
天池大赛nlp之sql


# haoojin
utils_wikisql.py
  get_g(sql_i)函数 
```python
g_sc.append( psql_i1["sel"])
g_sa.append( psql_i1["agg"])
-->
g_sc.append( psql_i1["sel"][0])
g_sa.append( psql_i1["agg"][0])
```
