import pandas as pd
import numpy as np
import pickle
import warnings
import os
import html

def read_item_index_to_entity_id_file():
    file = './data/item_index2entity_id_rehashed.txt'
    i = 0
    for line in open(file, encoding='utf-8').readlines():
        item_index = line.strip().split('\t')[0]
        satori_id = line.strip().split('\t')[1]
        item_index_old2new[item_index] = i
        entity_id2index[satori_id] = i        
        i += 1

# the knowledge graph is achieved from Microsoft Satori
def convert_rating_file():
    i = 0
    for line in open('./data/item_index2entity_id.txt', encoding='utf-8').readlines():
        context = line.strip().split('\t')
        item_index, entity_id = context[0], context[1]
        item_index_old2new[item_index] = i
        entity_id2index[entity_id] = i
        i += 1

    item_set = set(item_index_old2new.values())
    user_pos_ratings = dict()
    user_neg_ratings = dict()

    for line in open('./data/BX-Book-Ratings.csv', encoding='utf-8').readlines()[1:]:
        arr = line.strip().split(';')
        arr = list(map(lambda x: x[1:-1], arr))

        item_index_old = arr[1]
        if item_index_old not in item_index_old2new:  # the item is not in the final item set
            continue
        item_index = item_index_old2new[item_index_old]

        user_index_old = int(arr[0])
        rating = float(arr[2])
        
        if rating >= threshold:
            if user_index_old not in user_pos_ratings:
                user_pos_ratings[user_index_old] = set()
            user_pos_ratings[user_index_old].add(item_index)
        else:
            if user_index_old not in user_neg_ratings:
                user_neg_ratings[user_index_old] = set()
            user_neg_ratings[user_index_old].add(item_index)

    writer = open('./data/ratings_final.txt', 'w', encoding='utf-8')
    user_cnt = 0
    user_index_old2new = dict()
    for user_index_old, pos_item_set in user_pos_ratings.items():
        if user_index_old not in user_index_old2new:
            user_index_old2new[user_index_old] = user_cnt
            user_cnt += 1
        user_index = user_index_old2new[user_index_old]

        for item in pos_item_set:
            writer.write('%d\t%d\t1\n' % (user_index, item))
        unwatched_set = item_set - pos_item_set
        if user_index_old in user_neg_ratings:
            unwatched_set -= user_neg_ratings[user_index_old]
        for item in np.random.choice(list(unwatched_set), size=len(pos_item_set), replace=False):
            writer.write('%d\t%d\t0\n' % (user_index, item))
    writer.close()

def convert_KG_file():
    entity_cnt = len(entity_id2index)
    relation_cnt = 0

    writer = open('./data/kg_final.txt', 'w', encoding='utf-8')

    for line in open('./data/kg_rehashed.txt', encoding='utf-8').readlines():
        array = line.strip().split('\t')
        head_old = array[0]
        relation_old = array[1]
        tail_old = array[2]

        if head_old not in entity_id2index:
            entity_id2index[head_old] = entity_cnt
            entity_cnt += 1
        head = entity_id2index[head_old]

        if tail_old not in entity_id2index:
            entity_id2index[tail_old] = entity_cnt
            entity_cnt += 1
        tail = entity_id2index[tail_old]

        if relation_old not in relation_id2index:
            relation_id2index[relation_old] = relation_cnt
            relation_cnt += 1
        relation = relation_id2index[relation_old]

        writer.write('%d\t%d\t%d\n' % (head, relation, tail))

    writer.close()

if __name__ == '__main__':
    np.random.seed(555)

    threshold = 0
    entity_id2index = dict()
    relation_id2index = dict()
    item_index_old2new = dict()
    convert_rating_file()
    convert_KG_file()