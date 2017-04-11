import numpy as np
import itertools as it
import os
import re

ju = 0


def char_to_pos(c):
    if (c == ' '):
        return 26
    return ord(c) - 97

def pos_to_char(p):
    if (p == 26):
        return ' '
    return chr(p + 97)

def get_file(path):
    dirs = os.listdir(path)
    for d in dirs:
        with open (os.path.join(path, d), 'r') as myfile:
            txt = myfile.readlines()
        txt = ''.join(txt).replace('\n', '')
        yield txt 


def clean_txt(file_gen):
    for t in file_gen:
        words = t.split(' ')
        words = [w.lower() for w in words if w.isalpha()]
        valid_text = ' '.join(words)
        yield valid_text

def kgram_txt_prob(txt, trans):
    k = len(trans.shape) - 1
    def txt_to_kgram(txt, k):
        for i in range(k, len(txt)):
            step = list(range(k+1))[::-1]
            kgram = tuple([txt[i-s] for s in step])
            yield kgram
    for kgram in txt_to_kgram(txt, k):
        idx = tuple([char_to_pos(kg) for kg in kgram])
        if (idx[0] / 26 > 1 or idx[1] / 26 > 1 or idx[2] / 26 > 1):
            continue
        trans[idx] += 1
        global ju
        ju += 1
    


def process_txt(txt_gen, trans):
    clean = clean_txt(txt_gen)
    for t in clean:
       print('process new file') 
       kgram_txt_prob(t, trans)

def count_to_prob(mat):
    mat_prob = np.zeros(mat.shape)
    alph = list(range(27))
    idxs = it.product(alph, repeat=len(mat.shape)-1)
    for idx in idxs:
        s = mat[idx].sum()
        mat_prob[idx] = mat[idx] / s if s > 0 else  mat[idx]
    return mat_prob

def train_kgram_model(k, path):
    trans_dim = [27]*(k+1)
    trans_mat = np.zeros(trans_dim)
    
    F = get_file(path)
    process_txt(F, trans_mat)
    global ju
    print(ju)
    trans_mat_prob = count_to_prob(trans_mat)

    np.save('trans', trans_mat_prob)
    np.save('trans_countz', trans_mat)


#train_kgram_model(2, 'data/text')
#exit()

def pdf_to_cdf(mat, k):
    mat_cdf = np.zeros(mat.shape)
    alph = list(range(27))
    idxs = it.product(alph, repeat=k)
    for idx in idxs:
        mat_cdf[idx] = np.cumsum(mat[idx])
    return mat_cdf


def gen(seed, k, trans):
    cdf = trans
    sentence = '' + seed

    def nextletter(sentence, k):
        step = list(range(k))[::-1]
        kgram = tuple([sentence[i-s] for s in step])
        idx = tuple([char_to_pos(kg) for kg in kgram])
        sample = np.random.uniform(0,1)
        pos = np.argmax(trans[idx] > sample)
        #print(idx)
        #print(cdf[14][26])
        #print('__ ', pos)
        return pos_to_char(pos)

    while(True):
        c = nextletter(sentence, k)
        sentence += c
        yield sentence
        


t = np.load('trans.npy')

t_cdf = pdf_to_cdf(t, 2)

G = gen('comes', 2, t_cdf)

test = t_cdf[0,0]

s = ''

for i in range(100):
    s = next(G)

print(s)


