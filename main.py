import argparse
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
import numpy as np

from dataset import Dictionary, VQAFeatureDataset,load_dictionary
from train import train
import utils
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from models import FrameQA_model
from models import Count_model
from models import Trans_model
from models import Action_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--task', type=str, default='Action')
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='ParalCoAtt', help='temporalAtt, ParalCoAtt')
    parser.add_argument('--max_len',type=int, default=36)
    parser.add_argument('--output', type=str, default='saved_models/glimpse%d-subnum%d-reasonstep%d-lambda%s-cg-shuffle/%s')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1000, help='random seed')
    parser.add_argument('--sentense_file_path',type=str, default='data/dataset')
    parser.add_argument('--glove_file_path', type=str, default='/mnt/data2/lixiangpeng/models/glove/glove.6B.300d.txt')
    parser.add_argument('--feat_category',type=str,default='resnet')
    parser.add_argument('--feat_path',type=str,default='/mnt/data2/lixiangpeng/dataset/tgif/features')
    parser.add_argument('--Multi_Choice',type=int, default=5)
    parser.add_argument('--test_phase', type=bool, default=False)
    parser.add_argument('--scale', default=[256, 512, 1024])
    parser.add_argument('--reasonSteps', type=int, default=1)
    parser.add_argument('--sub_nums', type=int, default=8)
    parser.add_argument('--lambda', type=float, default=0.01)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    params = vars(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # dictionary = Dictionary.load_from_file('./dictionary.pkl')
    dictionary = load_dictionary(args.sentense_file_path, args.task)
    if not args.test_phase:
        train_dset = VQAFeatureDataset(args.task, dictionary, args.sentense_file_path,args.feat_category,args.feat_path, mode='Train')
        eval_dset = VQAFeatureDataset(args.task, dictionary, args.sentense_file_path,args.feat_category,args.feat_path, mode='Test')
        batch_size = args.batch_size

        model_name = args.task+'_model'
        model = getattr(locals()[model_name], 'build_%s' % args.model)(args.task, train_dset, params).cuda()
        # model.w_emb.init_embedding(dictionary,args.glove_file_path,args.task)

        print('========start train========')
        model = model.cuda()

        train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1)
        eval_loader = DataLoader(eval_dset, batch_size, shuffle=True, num_workers=1)
        train(model, train_loader, eval_loader, params)
    else:
        test_dset = VQAFeatureDataset(args.task, dictionary, args.sentense_file_path,args.feat_category,args.feat_path, mode='Valid')
        batch_size = args.batch_size
        model_name = args.task + '_model'
        model = getattr(locals()[model_name], 'build_%s' % args.model)(args.task, test_dset, args.num_hid).cuda()
        model.w_emb.init_embedding(dictionary, args.glove_file_path, args.task)
        print('========start test========')
        state_dict = torch.load(args.output % (len(params['scale']), params['sub_nums'], params['reasonSteps'], str(params['lambda']), params['task']) + '/model.pth')

        model.load_state_dict(state_dict)
        model = model.cuda()
        model.train(False)
        test_loader = DataLoader(test_dset, 1, shuffle=False, num_workers=1)
        test_score = model.sample(test_dset, test_loader)
        # print('test score: %.2f ' % (100 * test_score))
