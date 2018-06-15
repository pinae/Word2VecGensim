#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
import gensim
import logging
import os


class MySentences(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in open(self.filename):
            yield line.split()


sentences = MySentences(os.path.join("de_wiki_text", "de", "full.txt"))
logging.basicConfig(format='%(message)s', level=logging.INFO)
model = gensim.models.Word2Vec(sentences, size=200, sg=1, negative=0, hs=1, workers=8, iter=5)
model.save('gensim_model.mod')
