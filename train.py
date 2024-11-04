import logging
import os.path, sys, pdb, re
from collections import deque
from tqdm import tqdm

import click
import numpy as np
import torch
import torch.optim as optim

from livdet.data.data_loader import get_fcn_dataset as get_dataset
from livdet.data.util import get_data_ids
from livdet.models import get_model
from livdet.models.models import models
from livdet.models.util import get_scale_label, wbce_loss
from livdet.transforms import augment_collate
from livdet.util import config_logging
from livdet.tools.util import display_loss, display_loss_multiline
from livdet.torch_utils import to_device, make_variable

def roundrobin_infinite(*loaders):
    if not loaders:
        return
    iters = [iter(loader) for loader in loaders]
    while True:
        for i in range(len(iters)):
            it = iters[i]
            try:
                yield next(it)
            except StopIteration:
                iters[i] = iter(loaders[i])
                yield next(iters[i])

@click.command()
@click.argument('output')
@click.option('--dataset', required=True, multiple=True)
@click.option('--validdata', default="")
@click.option('--datadir', default="", type=click.Path(exists=True))
@click.option('--batch_size', '-b', default=1)
@click.option('--lr', '-l', default=0.001)
@click.option('--step', type=int)
@click.option('--iterations', '-i', default=100000)
@click.option('--momentum', '-m', default=0.9)
@click.option('--snapshot', '-s', default=5000)
@click.option('--downscale', type=int)
@click.option('--augmentation/--no-augmentation', default=False)
@click.option('--fyu/--torch', default=False)
@click.option('--crop_size', default=720)
@click.option('--weights', type=click.Path(exists=True))
@click.option('--model', default='frcn', type=click.Choice(models.keys()))
@click.option('--loss_criterion', default="wbce")
@click.option('--gpu', default='0')
@click.option('--use_validation/--no-use_validation', default=False)
@click.option('--normalize', default='minmax_image', type=str)
@click.option('--model3d/--no-model3d', default=True)
def main(output, dataset, validdata, datadir, batch_size, lr, step, iterations,
        momentum, snapshot, downscale, augmentation, use_validation, fyu, crop_size,
        weights, model, gpu, normalize, loss_criterion, model3d):
    if os.path.exists('{}-best_EMA.pth'.format(output)):
        raise IOError('There is a file named ' + '{}-best_EMA.pth'.format(output))

    traindatasets, validdatasets, _ = get_data_ids(dataset[0])
    num_cls = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    config_logging()

    net = get_model(model, num_cls=num_cls, finetune=True)
    net.cuda()
    if weights is not None:
        weights = np.loadtxt(weights)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, nesterov = True, weight_decay=1e-06)
    step_size = 20000
    scheduler = optim.lr_scheduler.StepLR(opt, step_size, gamma=0.1)

    transform = []
    target_transform = []
    label_type = 'lesion'
    data_folder = re.split('[1-9]',dataset[0])[0]
    print('dataset: ', data_folder)
    datasets = [get_dataset(dataset[0], os.path.join(datadir, data_folder), datasets=traindatasets, \
                mode='train', normalize=normalize, crop_size=crop_size, label_type=label_type)]
    if augmentation:
        collate_fn = lambda batch: augment_collate(batch, crop=crop_size, flip=True)
    else:
        collate_fn = torch.utils.data.dataloader.default_collate
    loaders = [torch.utils.data.DataLoader(data, batch_size=batch_size,
                                           shuffle=True, num_workers=2,
                                           collate_fn=collate_fn,
                                           pin_memory=True)
               for data in datasets]

    if use_validation:
        validdata_folder = re.split('[1-9]',dataset[0])[0]
        print('validation dataset: ', validdata_folder)
        valid_data = get_dataset(dataset[0], os.path.join(datadir, validdata_folder), datasets=validdatasets, \
                        mode='validation', normalize=normalize, crop_size=crop_size, label_type=label_type)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=4)
        valid_data_size = len(valid_data)

    validfreq = 200
    showlossfreq = 500
    savefre = 500
    best_score = 1e9
    best_score_EMA = 1e9
    count_ = 0
    tolerance = 100
    iteration = 0
    losses = deque(maxlen=10)
    params = dict()
    params['scale'] = 1.0
    lesion_weight = 10.0

    lesion_vals = []
    steps, vals = [], []
    valid_steps, valid_vals = [], []
    valid_lesion_vals = []
    valid_vals_EMA, coef_EMA, count_EMA = [], 0.9, 1

    for im, *label in roundrobin_infinite(*loaders):
        # Save checkpoints
        if np.mod(iteration, savefre) == 0:
            torch.save(net.state_dict(), '{}.pth'.format(output))
            print('Save weights to: ', '{}.pth'.format(output))
        if iteration % snapshot == 0:
            torch.save(net.state_dict(), '{}-iter{}.pth'.format(output, iteration))
        # log results
        if iteration == 0:
            logging.info('Iteration {}:\t{}'.format(iteration, float('inf')))
        elif iteration % 100 == 0:
            logging.info('Iteration {}:\t{}'.format(iteration, np.mean(losses)))

        # Evaluation on validation dataset
        with torch.no_grad():
            if use_validation and iteration % validfreq == 0:
                running_valid_loss_lesion, running_valid_loss_liver = 0.0, 0.0
                running_valid_loss = 0.0
                for valid_im, *valid_label in valid_loader:
                    if isinstance(valid_label, list):
                        if len(valid_label) == 1:
                            valid_label = valid_label[0]
                        elif len(valid_label) == 2:
                            raise ValueError('Error in data loading.')

                    # Lesion loss on validatation set
                    valid_pred_lesion, *valid_pred_context = net.predict(valid_im)
                    valid_label_scale = get_scale_label(valid_label, params)
                    if loss_criterion == 'wbce':
                        valid_loss_lesion = wbce_loss(valid_label_scale, torch.from_numpy(valid_pred_lesion),
                                        None, weight=lesion_weight, model3D=model3d)

                    running_valid_loss_lesion += valid_loss_lesion.numpy().mean() * valid_im.size(0)


                    # Conext encoding loss on validatation set
                    valid_loss_context5 = wbce_loss(valid_label_scale, torch.from_numpy(valid_pred_context[0]), None,\
                                    weight=lesion_weight, model3D=model3d)
                    valid_loss_context4 = wbce_loss(valid_label_scale, torch.from_numpy(valid_pred_context[1]), None,\
                                    weight=lesion_weight, model3D=model3d)
                    valid_loss_context3 = wbce_loss(valid_label_scale, torch.from_numpy(valid_pred_context[2]), None,\
                                    weight=lesion_weight, model3D=model3d)
                    valid_loss_context2 = wbce_loss(valid_label_scale, torch.from_numpy(valid_pred_context[3]), None,\
                                    weight=lesion_weight, model3D=model3d)
                    valid_loss_context1 = wbce_loss(valid_label_scale, torch.from_numpy(valid_pred_context[4]), None,\
                                    weight=lesion_weight, model3D=model3d)
                    valid_loss_context0 = torch.zeros(1)
                    valid_loss_context = valid_loss_context5 + valid_loss_context4 + valid_loss_context3 + valid_loss_context2 + valid_loss_context1

                    running_valid_loss += valid_loss_lesion.numpy().mean() * valid_im.size(0) # do not use context loss for early stop

                running_valid_loss_lesion = running_valid_loss_lesion / valid_data_size
                running_valid_loss = running_valid_loss / valid_data_size
                valid_steps.append(iteration)
                valid_lesion_vals.append(running_valid_loss_lesion)
                valid_vals.append(running_valid_loss)

                if not valid_vals_EMA: # exponential moving weighted average
                    valid_val_cur = 0.0
                valid_val_cur = coef_EMA * valid_val_cur + (1 - coef_EMA) * running_valid_loss
                valid_val_cur_biascorr = valid_val_cur / (1 - coef_EMA ** count_EMA)
                valid_vals_EMA.append(valid_val_cur_biascorr)
                count_EMA += 1

                print('\nValidation loss-EMA: {0}, best score-EMA: {1}'.format(valid_val_cur_biascorr, best_score_EMA))
                if valid_val_cur_biascorr <=  best_score_EMA:
                    best_score_EMA = valid_val_cur_biascorr
                    print('update to new best_score_EMA: {}'.format(best_score_EMA))
                    torch.save(net.state_dict(), '{}-best_EMA.pth'.format(output))
                    print('Save best weights to: ', '{}-best_EMA.pth'.format(output))
                    count_ = 0
                else:
                    count_ = count_ + 1
                print('\nValidation loss: {}, best_score: {}'.format(running_valid_loss, best_score))
                if running_valid_loss <=  best_score:
                    best_score = running_valid_loss
                    print('update to new best_score: {}'.format(best_score))
                    torch.save(net.state_dict(), '{}-best.pth'.format(output))
                    print('Save best weights to: ', '{}-best.pth'.format(output))
                if count_ >= tolerance:
                    torch.save(net.state_dict(), '{}-iter{}.pth'.format(output, iteration)) # Save final-iteration model
                    torch.save(net.state_dict(), '{}.pth'.format(output))
                    assert 0, 'performance not imporoved for so long'

        net.train()
        opt.zero_grad()
        if isinstance(label, list):
            if len(label) == 1:
                label = label[0]
            elif len(label) == 2:
                raise ValueError('Error in data loading.')
        im_v = make_variable(im, requires_grad=False)
        label_scale = get_scale_label(label, params)
        label_scale = make_variable(label_scale, requires_grad=False)

        preds, *pred_context = net(im_v)
        if loss_criterion == 'wbce':
            loss_lesion = wbce_loss(label_scale, preds, None, weight=lesion_weight, model3D=model3d)
        loss_lesion_value = loss_lesion.data.cpu().numpy().mean()
        assert not np.isnan(loss_lesion_value), "nan error in lesion detection loss"

        # Context ecnoding loss for lesion detection
        loss_context5 = wbce_loss(label_scale, pred_context[0], None, weight=lesion_weight, model3D=model3d)
        loss_context4 = wbce_loss(label_scale, pred_context[1], None, weight=lesion_weight, model3D=model3d)
        loss_context3 = wbce_loss(label_scale, pred_context[2], None, weight=lesion_weight, model3D=model3d)
        loss_context2 = wbce_loss(label_scale, pred_context[3], None, weight=lesion_weight, model3D=model3d)
        loss_context1 = wbce_loss(label_scale, pred_context[4], None, weight=lesion_weight, model3D=model3d)
        loss_context0 = torch.zeros(1)
        loss_context = loss_context5 + loss_context4 + loss_context3 + loss_context2 + loss_context1

        # Joint loss
        loss = loss_lesion + loss_context
        loss_value = loss.data.cpu().numpy().mean()
        assert not np.isnan(loss_value) ,"nan error in joint loss"

        steps.append(iteration)
        lesion_vals.append(loss_lesion_value)
        vals.append(loss_value)

        loss.backward()
        losses.append(loss_value)
        opt.step()

        iteration += 1

        # Plot loss curves
        if iteration % showlossfreq == 0:
            display_loss(steps, vals, plot=None, name = dataset[0] + '-' + model + '-context-total', legend='context-total')
            display_loss(steps, lesion_vals, plot=None, name=dataset[0] + '-' + model+ '-context-lesion', legend='lesion')
            if use_validation:
                display_loss_multiline(valid_steps, [valid_vals, valid_vals_EMA], plot=None, name=dataset[0] + '-' + model + '-valid-context-total', legend=['original', 'EMA'])

        scheduler.step() # learning rate decay every step_size iterations
        if iteration >= iterations:
            torch.save(net.state_dict(), '{}-iter{}.pth'.format(output, iteration)) # Save final-iteration model
            torch.save(net.state_dict(), '{}.pth'.format(output))
            logging.info('Optimization complete.')
            break

if __name__ == '__main__':
    main()
