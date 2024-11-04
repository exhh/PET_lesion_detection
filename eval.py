import os, sys, re
import time
from tqdm import *

import click
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
from scipy.special import expit, logit

from livdet.data.data_loader import dataset_obj
from livdet.data.data_loader import get_fcn_dataset
from livdet.data.util import get_data_ids
from livdet.models.models import get_model
from livdet.models.models import models
from livdet.util import to_tensor_raw

import scipy.io as sio
import copy

@click.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--train_val_test', required=True, multiple=True)
@click.option('--dataset', default='liver3D')
@click.option('--datadir', default='',
        type=click.Path(exists=True))
@click.option('--eval_result_folder', default='experiments',
        type=click.Path(exists=True))
@click.option('--model', default='frcn', type=click.Choice(models.keys()))
@click.option('--loss_criterion', default='wbce')
@click.option('--gpu', default='0')
@click.option('--normalize', default='minmax_image', type=str)
@click.option('--eval2d_3d', required=True, multiple=True)
@click.option('--predict_type', default='fuse')
@click.option('--qual_eval/--no_qual_eval', default=False)
@click.option('--num_codeword', default=16)
@click.option('--lesion_weight', default=10.0)
@click.option('--eval_det/--no_eval_det', default=False)
@click.option('--eval_seg/--no_eval_seg', default=False)
def main(path, train_val_test, dataset, datadir, eval_result_folder, model, loss_criterion,\
        gpu, normalize, eval2d_3d, predict_type, qual_eval, num_codeword, lesion_weight,\
        eval_det, eval_seg):
    models_lesion = ('spaounet3dall', )
    models_spatial_context = ('spaounet3dall', )
    traindatasets, validdatasets, testdatasets = get_data_ids(dataset)
    print('GPU ' + gpu)
    print('Model ' + model)
    print('Test ' + dataset)
    print('Test data = ', testdatasets)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    if os.path.isfile(path):
        print('Evaluate model: ', path)
    else:
        print('No trained model!')
    print('loss_criterion: ', loss_criterion)
    print('Normalize: ', normalize)
    print('eval2d_3d: ', eval2d_3d)
    print('num_codeword: ', num_codeword)
    print('lesion_weight: ', lesion_weight)
    print('eval_det: ', eval_det)
    print('eval_seg: ', eval_seg)
    if model in models_spatial_context:
        if predict_type == 'nofuse':
            predict_type = ''
        else:
            predict_type = '-' + predict_type
    else:
        predict_type = ''
    print('predict_type: ', predict_type)
    print('qual_eval: ', qual_eval)

    label_type = 'lesion'
    data_split_dict = {'test':testdatasets, 'val':validdatasets, 'train':traindatasets}
    data_folder = re.split('[1-9]',dataset)[0]
    for data_split in train_val_test:
        print('Evaluate ' + data_split + ' of ' + dataset)
        ds = get_fcn_dataset(dataset, os.path.join(datadir,data_folder), datasets=testdatasets,\
            mode='test', normalize=normalize, label_type=label_type)

        num_cls = 1
        local_min_len = 1
        avg_filter_size = 2 * local_min_len + 1
        iou_pool = (0.0, 0.05, 0.1)
        use_sigmoid = '-sigmoid' #

        loader = torch.utils.data.DataLoader(ds, num_workers=8)
        if len(loader) == 0:
            print('Empty data loader')
            return
        else:
            train_model_path = path.split('/',2)[1] + '_' + path.rsplit('/',1)[1].split('.')[0]
            savefolder = os.path.join(eval_result_folder, data_split, dataset, train_model_path + use_sigmoid + predict_type)
            if not os.path.exists(savefolder):
                os.makedirs(savefolder)

        # load model
        net = get_model(model, num_cls=num_cls, num_codeword=num_codeword)
        weights_dict = torch.load(path, map_location=lambda storage, loc: storage)
        net.load_state_dict(weights_dict)
        net.eval()

        resultsDict = {}
        if model in models_lesion:
            area_pool = [5, 10, 20]
            votingmap_name = 'votingmap_lesion'
        voting_time_name = 'prediction_time'
        mask_name = 'mask'
        threshold_pool = np.arange(0.0, 1.01, 0.05)
        threshold_pool = np.insert(threshold_pool, 1, 0.01) # insert 0.01 between 0 and 0.05

        with torch.no_grad():
            iterations = tqdm(enumerate(loader))
            for im_i, (im, *iminfo) in iterations: # one subject at a time
                if isinstance(iminfo, list):
                    if len(iminfo) == 1:
                        raise Exception("The file information format is not correct.")
                    elif len(iminfo) == 2:
                        im_names = iminfo[1] # all the images names for one subject
                        subject_name = iminfo[0][0]
                        print('subject name: ', subject_name)

                im = Variable(im.cuda())
                votingStarting_time = time.time()
                VotingMap, *pred_auxi = net.predict(im) # VotingMap for lesion
                if model in models_spatial_context:
                    if not predict_type:
                        VotingMap = pred_auxi[0]
                    elif predict_type == '-ensem':
                        VotingMap = logit((expit(VotingMap) + expit(pred_auxi[0])) / 2.0)

                votingEnding_time = time.time()
                resultsDict[voting_time_name] = votingEnding_time - votingStarting_time
                print("prediction time: ", votingEnding_time - votingStarting_time)
                if isinstance(VotingMap, list):
                    VotingMap = VotingMap[0]
                VotingMap = np.squeeze(VotingMap)
                np.save(os.path.join(savefolder, subject_name + '_prediction_map.npy'), VotingMap)

if __name__ == '__main__':
    main()
