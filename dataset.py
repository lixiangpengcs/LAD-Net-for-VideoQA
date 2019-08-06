from __future__ import print_function
import os
import json
import pickle as pkl
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
import utils
import pandas as pd
import time
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

load_csv = {'Count':'Count','Trans':'Transition','FrameQA':'FrameQA','Action':'Action'}
mc_task = ['trans','action']

def get_captions(row, task):
    if task.lower() in mc_task:
        columns = ['question', 'a1', 'a2', 'a3', 'a4', 'a5']
    else:
        columns = ['question']
    sents = [row[1][col] for col in columns if not pd.isnull(row[1][col])]
    return sents


def get_ques_pairs(row, task):
    if task.lower() in mc_task:
        columns = ['question', 'answer', 'key', 'a1', 'a2', 'a3', 'a4', 'a5']
    else:
        columns = ['question','answer','key']
    sents = [row[1][col] for col in columns if not pd.isnull(row[1][col])]
    return sents



class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx[w])
        return tokens

    def dump_to_file(self, path):
        pkl.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = pkl.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

def filter_answers(answers_dset):
    """This will change the answer to preprocessed version
    """
    occurence = []

    for ans_entry in answers_dset:
        if ans_entry not in occurence:
            occurence.append(ans_entry)
    return occurence

def create_ans2label(occurence, name, cache_root='data/cache'):
    """Note that this will also create label2ans.pkl at the same time

    occurence: dict {answer -> whatever}
    name: prefix of the output file
    cache_root: str
    """
    ans2label = {}
    label2ans = []
    label = 0
    for answer in occurence:
        label2ans.append(answer)
        ans2label[answer] = label
        label += 1

    utils.create_dir(cache_root)

    cache_file = os.path.join(cache_root, name+'_ans2label.pkl')
    pkl.dump(ans2label, open(cache_file, 'wb'))
    cache_file = os.path.join(cache_root, name+'_label2ans.pkl')
    pkl.dump(label2ans, open(cache_file, 'wb'))
    return ans2label, label2ans

def load_dictionary(load_path, task):

    if os.path.exists('./data/%s_dict.pkl'%task.lower()):
        dictionary = Dictionary.load_from_file(os.path.join('./data/%s_dict.pkl')%task.lower())
        print('loading dictionary done.')
    else:
        print('Creating %s dictionary...'%task)
        file = os.path.join(load_path,'Total_%s_question.csv'%load_csv[task].lower())
        total_q = pd.DataFrame().from_csv(file, sep='\t')
        all_sents = []
        dictionary = Dictionary()
        for row in total_q.iterrows():
            all_sents.extend(get_captions(row, task))
        for q in all_sents:
            dictionary.tokenize(q, True)

        dictionary.dump_to_file('./data/%s_dict.pkl'%task.lower())
    return dictionary

def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = map(float, vals[1:])
        word2emb[word] = np.array(vals)
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb

def get_q_a_v_pair(file, ans2label, name,task, cache_root='data/cache'):
    total_q = pd.DataFrame().from_csv(file, sep='\t')

    target = []
    for i,row in enumerate(total_q.iterrows()):
        q_a_v = get_ques_pairs(row, task)
        target.append({
            'id': i,
            'question': q_a_v[0],
            'answer':q_a_v[1],
            'key':q_a_v[2],
            'label':ans2label[q_a_v[1]]
        })
    # print( target)
    utils.create_dir(cache_root)
    cache_file = os.path.join(cache_root, name + '_target.pkl')
    pkl.dump(target, open(cache_file, 'wb'))
    return target

def get_q_cand_a_v_pair(file,ans2label, name, task, cache_root='data/cache'):
    total_q = pd.DataFrame().from_csv(file, sep='\t')
    target = []
    print('Constructing %s target...'%name)
    for i, row in enumerate(total_q.iterrows()):
        q_a_v = get_ques_pairs(row, task)
        target.append({
            'id': i,
            'question': q_a_v[0],
            'answer': q_a_v[1],
            'key': q_a_v[2],
            'a1':q_a_v[3],
            'a2':q_a_v[4],
            'a3':q_a_v[5],
            'a4':q_a_v[6],
            'a5':q_a_v[7],
            'label': q_a_v[1]
        })

    utils.create_dir(cache_root)
    cache_file = os.path.join(cache_root, name + '_target.pkl')
    pkl.dump(target, open(cache_file, 'wb'))
    return target


def _load_dataset(dataroot, mode,task, ans2label):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    print('Loading %s  %s dataset ...'%(mode, task))
    answer_path = os.path.join(dataroot, 'cache', '%s_%s_target.pkl' % (mode,task.lower()))  #Train_frameqa
    if os.path.exists(answer_path):
        answers = pkl.load(open(answer_path, 'rb'))
        entries = sorted(answers, key=lambda x: x['id'])
    else:
        file = 'data/dataset/%s_%s_question.csv' % (mode, load_csv[task].lower())
        if task.lower() in mc_task:
            entries = get_q_cand_a_v_pair(file, ans2label, name=mode + '_' + task.lower(), task=task)
        else:
            entries = get_q_a_v_pair(file, ans2label, name=mode+'_'+task.lower(), task=task)
    return entries


class VQAFeatureDataset(Dataset):
    def __init__(self, task, dictionary, dataroot='data', feat_category='resnet', feat_path='./data/',mode='Train', Mul_Choice=5):
        super(VQAFeatureDataset, self).__init__()
        #assert name in ['Train_frameqa', 'Test_frameqa']

        self.feat_category = feat_category
        self.feat_path = feat_path
        self.task = task
        self.Mul_Choice = Mul_Choice
        f = os.path.join('./data', 'cache', '%s_ans2label.pkl'%task)
        if not os.path.exists(f):
            print('Constructing ans2label ...')
            file = os.path.join(dataroot,'Total_%s_question.csv'%load_csv[task].lower())
            total_q = pd.DataFrame().from_csv(file, sep='\t')['answer']
            occurence = filter_answers(total_q)
            self.ans2label, self.label2ans = create_ans2label(occurence, task)
        else:
            ans2label_path = os.path.join('./data', 'cache', '%s_ans2label.pkl'%task)
            label2ans_path = os.path.join('./data', 'cache', '%s_label2ans.pkl'%task)
            self.ans2label = pkl.load(open(ans2label_path, 'rb'))
            self.label2ans = pkl.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary
        self.max_length = 36
        if os.path.exists('data/%s_%s_entries_info.pkl'%(task, mode)):
            self.entries = pkl.load(open('data/%s_%s_entries_info.pkl'%(task, mode), 'rb'))
        else:
            self.entries = _load_dataset('./data', mode, task, self.ans2label)
            self.tokenize()
            with open('data/%s_%s_entries_info.pkl'%(task, mode), 'wb') as f:
                pkl.dump(self.entries, f)

        self.tensorize()
        if self.feat_category.lower()=='resnet':
            self.v_dim = 2048
        elif self.feat_category.lower()=='c3d':
            self.v_dim = 4096
        else:
            raise ValueError('The feature you used raise error!!!')

    def _load_video(self, index):
        self.features = h5py.File(os.path.join(self.feat_path, 'TGIF_%s_pool5.hdf5'%self.feat_category.upper()), 'r')

        feature = self.features[str(index)][:].astype('float32')
        feature = utils.pad_video(feature, (self.max_length, self.v_dim)).astype('float32')
        # shuffle index
        import random
        idxs = list(range(0,36))
        random.shuffle(idxs)
        feature = feature[idxs]
        return torch.from_numpy(feature)

    def tokenize(self, max_length=35):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        # if self.task.lower() not in mc_task:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        model.eval().to('cuda')
        if self.task.lower() in mc_task:
            for entry in self.entries:
                tokens = []
                q_embeds = []
                for i, candi in enumerate(['a1','a2','a3','a4','a5']):
                    # for bert   [CLS]  question # [SEP] option [SEP]
                    ques = entry['question']
                    option = entry[candi]
                    tokenized_ques = tokenizer.tokenize(ques)
                    tokenized_opt = tokenizer.tokenize(option)
                    tokenized_ques = tokenized_ques + tokenized_opt
                    if len(tokenized_ques) > max_length-2:
                        tokenized_ques = tokenized_ques[0:(max_length - 2)]  # account for [CLS] and [SEP]
                    tokens_bert = ['[CLS]'] + tokenized_ques + ['[SEP]']
                    indexed_tokens = tokenizer.convert_tokens_to_ids(tokens_bert)
                    input_mask = [1] * len(indexed_tokens)
                    while len(indexed_tokens) < max_length:
                        indexed_tokens.append(0)
                        input_mask.append(0)
                    token_tensor = torch.tensor([indexed_tokens]).to('cuda')
                    mask = torch.tensor([input_mask]).to('cuda')
                    with torch.no_grad():
                        encoded_layers, _ = model(token_tensor, mask)
                        q_embeds.append(np.array(encoded_layers[-1].squeeze()))

                    token = self.dictionary.tokenize(entry['question'], False)
                    token_candi = self.dictionary.tokenize(entry[candi], False)
                    token = (token + token_candi)[:max_length]
                    if len(token) < max_length:
                        # Note here we pad in front of the sentence
                        padding = [self.dictionary.padding_idx] * (max_length - len(token))
                        token = padding + token
                    utils.assert_eq(len(token), max_length)
                    tokens.append(token)
                entry['q_token'] = np.array(tokens)
                entry['q_embed'] = np.array(q_embeds)
                assert entry['q_embed'].shape == (5, 35, 768)
                assert entry['q_token'].shape[-1] == max_length
        else:
            for entry in self.entries:
                ques = entry['question']
                tokenized_ques = tokenizer.tokenize(ques)
                if len(tokenized_ques) > max_length-2:
                    tokenized_ques = tokenized_ques[0:(max_length - 2)]  # account for [CLS] and [SEP]
                tokens_bert = []
                input_type_ids = []
                tokens_bert.append("[CLS]")
                input_type_ids.append(0)
                for tok in tokenized_ques:
                    tokens_bert.append(tok)
                    input_type_ids.append(0)
                tokens_bert.append('[SEP]')
                input_type_ids.append(0)
                indexed_tokens = tokenizer.convert_tokens_to_ids(tokens_bert)
                input_mask = [1] * len(indexed_tokens)
                while len(indexed_tokens) < max_length:
                    indexed_tokens.append(0)
                    input_mask.append(0)
                    input_type_ids.append(0)
                token_tensor = torch.tensor([indexed_tokens]).to('cuda')
                mask = torch.tensor([input_mask]).to('cuda')
                with torch.no_grad():
                    encoded_layers, _ = model(token_tensor, mask)
                    entry['q_embed'] = np.array(encoded_layers[-1].squeeze())

                tokens = self.dictionary.tokenize(entry['question'], False)
                tokens = tokens[:max_length]
                entry['q_mask'] = np.zeros(max_length)
                entry['q_mask'][-len(tokens):] = 1
                if len(tokens) < max_length:
                    # Note here we pad in front of the sentence
                    padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                    tokens = padding + tokens
                utils.assert_eq(len(tokens), max_length)
                entry['q_token'] = tokens



    def tensorize(self):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question
            ques_embed = torch.from_numpy(entry['q_embed'])
            entry['q_embed'] = ques_embed
            labels= []
            if self.task.lower() =='count':
                labels.append(max(entry['answer'], 1))
            else:
                labels.append(entry['label'])
            labels = np.array(labels)
            labels = torch.from_numpy(labels)
            entry['labels'] = labels


    def __getitem__(self, index):
        entry = self.entries[index]
        q_embed = entry['q_embed']
        features = self._load_video(entry['key'])
        question = entry['q_token']
        labels = entry['labels']
        return features, question, q_embed, labels[0]

    def __len__(self):
        return len(self.entries)


