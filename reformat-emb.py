#!/usr/bin/python

import sys


emb_old = open('En_vectors.txt', 'r').readlines()

new = open('embeddings.w2v.txt', 'w')

for x in emb_old:
    # testing to see if first params contain comma
    d = x.split(' ')
    dataWrite = x.split(',')
    if d[0].find(',') != -1:
        dat = d[1].split(',')
        dat.insert(0, d[0])
        dataWrite = dat
    new.write(" ".join(dataWrite))


