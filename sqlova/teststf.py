# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 19:20:25 2019

@author: 63184
"""

from stanza.nlp.corenlp import CoreNLPClient
client = CoreNLPClient(server='http://localhost:20000', default_annotators=['ssplit', 'lemma', 'tokenize', 'pos', 'ner']) # 注意在以前的版本中，中文分词为 segment，新版已经和其他语言统一为 tokenize

# 分词和词性标注测试
test1 = "深蓝的天空中挂着一轮金黄的圆月，下面是海边的沙地，都种着一望无际的碧绿的西瓜，其间有一个十一二岁的少年，项带银圈，手捏一柄钢叉，向一匹猹尽力的刺去，那猹却将身一扭，反从他的胯下逃走了。"
annotated = client.annotate(test1)
for sentence in annotated.sentences:
    for token in sentence:
        print(token.word, token.pos)

# 命名实体识别测试
test2 = "大概是物以希为贵罢。北京的白菜运往浙江，便用红头绳系住菜根，倒挂在水果店头，尊为胶菜；福建野生着的芦荟，一到北京就请进温室，且美其名曰龙舌兰。我到仙台也颇受了这样的优待……"
annotated = client.annotate(test2)
for sentence in annotated.sentences:
    for token in sentence:
        if token.ner != 'O':
          print(token.word, token.ner)