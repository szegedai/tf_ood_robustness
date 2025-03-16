import collections
import glob
from argparse import ArgumentParser

import numpy as np
import tqdm
from torch.utils.tensorboard import SummaryWriter

from datareader import ZengData
from models_pth import TFModel, WideResNet, multi_attention_resnet
import os
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR, CyclicLR, CosineAnnealingLR, CosineAnnealingWarmRestarts


def main(params):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(params)
    tl = collections.OrderedDict()
    vl = collections.OrderedDict()
    if len(params.fname) == 1:
        params.fname = glob.glob(params.fname[0])
        print('Number of found npz files: ', len(params.fname))
    for fn in params.fname:
        ds = ZengData(f_path=fn, binary_label=True, flip_chanels=True)
        x_train, y_train = ds.get_train()
        x_val, y_val = ds.get_val()
        print(f'Tf: {ds.get_name()}', x_train.shape, x_val.shape, y_train.shape, y_val.shape)
        training_loader = torch.utils.data.DataLoader(TensorDataset(torch.as_tensor(x_train), torch.as_tensor(y_train)),
                                                      batch_size=params.batch_size // len(params.fname), shuffle=True)
        tli = tl.get(ds.get_name(), [])
        tli.append(training_loader)
        tl[ds.get_name()] = tli
        validation_loader = torch.utils.data.DataLoader(TensorDataset(torch.as_tensor(x_val), torch.as_tensor(y_val)),
                                                        batch_size=params.batch_size,
                                                        shuffle=False)
        vli = vl.get(ds.get_name(), [])
        vli.append(validation_loader)
        vl[ds.get_name()] = vli
        print(len(training_loader))
        print(len(validation_loader))

    if params.m_type == "tf":
        model = TFModel()
    elif params.m_type == "wrn":
        model = WideResNet(params.d, len(tl), params.w, 0)

    model.to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=.9, nesterov=True, weight_decay=1e-5)
    if params.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.wd)
    elif params.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=params.lr, momentum=.9, nesterov=True,
                                    weight_decay=params.wd)
    else:
        raise Exception(f'Unsupported opt: {params.opt}')
    if params.scheduler == 'step':
        scheduler = MultiStepLR(optimizer,
                                milestones=[int(params.iter * .3), int(params.iter * .6), int(params.iter * .8)],
                                gamma=0.2)
    elif params.scheduler == 'cyclic':
        scheduler = CyclicLR(optimizer, max_lr=params.lr, base_lr=0,
                             step_size_up=int(params.iter * .1),
                             step_size_down=int(params.iter * .1))
    elif params.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, eta_min=0, T_max=params.iter)
    elif params.scheduler == 'cosine_restart':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=788, T_mult=2, eta_min=0)
    else:
        raise Exception(f'Unknown scheduler:{params.scheduler}')

    out_dir = params.save_dir
    writer = SummaryWriter(out_dir)

    best_vloss = 1_000_000.
    best_vacc = 0

    def generator(loader):
        while True:
            for data in loader:
                # Every data instance is an input + label pair
                # inputs, labels = data
                yield data

    dgs = collections.OrderedDict()
    for k in tl.keys():
        for tli in tl[k]:
            dg = generator(tli)
            dgi = dgs.get(k, [])
            dgi.append(dg)
            dgs[k] = dgi

    running_loss = None
    runing_acc = None
    ema_weight = .9
    for i in tqdm.tqdm(range(params.iter)):
        loss = None
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)

        # Zero your gradients for every batch!
        optimizer.zero_grad()
        input_list, lab_list, idxs = [], [], []
        for idx, k in enumerate(dgs.keys()):
            for dgi in dgs[k]:
                # Every data instance is an input + label pair
                data = next(dgi)
                input_list.append(data[0].to(device))
                lab_list.append(data[1].to(device))
                idxs.append(idx)

        inputs = torch.cat(input_list, 0).to(torch.float32)
        # Make predictions for this batch
        outputs = model(inputs)
        # inputs, labels = data[0].to(device), data[1].to(device)
        j0 = 0
        l_comps = 0.
        n_correct, n = 0, 0
        for j, idx in enumerate(idxs):
            labels = lab_list[j].to(torch.float32)
            outputj = outputs[j0:(j0 + labels.size()[0]), idx:(idx + 1)]
            j0 += labels.size()[0]
            pred = outputj >= 0 + 0
            n_correcti = torch.sum(pred == labels)
            n += outputj.size()[0]
            n_correct += n_correcti

            # Compute the loss and its gradients
            li = loss_fn(outputj, labels)
            if loss is None:
                loss = li
            else:
                loss += li
            l_comps += 1.

        loss /= l_comps
        acci = n_correct / n
        if runing_acc is None:
            runing_acc = acci
            running_loss = loss
        else:
            runing_acc = runing_acc * (1 - ema_weight) + ema_weight * acci
            running_loss = running_loss * (1 - ema_weight) + ema_weight * loss.item()
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        scheduler.step()

        # Gather data and report
        if i % params.eval_freq == (params.eval_freq - 1):
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            model.eval()

            # Disable gradient computation and reduce memory consumption.
            val_acc, avg_vloss, c = 0, 0, 0
            with torch.no_grad():
                for idx, k in enumerate(vl.keys()):
                    for vli in vl[k]:
                        c += 1
                        n_correcti, n, vlossi = 0, 0, 0
                        for vdata in vli:
                            vinputs, vlabels = vdata[0].to(torch.float32).to(device), vdata[1].to(torch.float32).to(device)
                            voutputs = model(vinputs)[:, idx:(idx + 1)]
                            n_correcti += torch.sum((voutputs >= 0 + 0) == vlabels)
                            vloss = loss_fn(voutputs, vlabels)
                            vlossi += vloss.item() * vinputs.size()[0]
                            n += vinputs.size()[0]
                        avg_vloss += vlossi / n
                        val_acc += n_correcti / n

            avg_vloss = avg_vloss / c
            val_acc = val_acc / c
            # print(                'LOSS/ACC train {:.4f}/{:.4f} valid {:.4f}/{:.4f}'.format(running_loss, runing_acc, avg_vloss, val_acc))
            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars('Loss', {'train': running_loss, 'val': avg_vloss}, i)
            writer.add_scalars('Acc', {'train': runing_acc, 'val': val_acc}, i)

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                print(f'Best vloss model saving: {best_vloss}-->{avg_vloss}')
                best_vloss = avg_vloss
                model_path = os.path.join(out_dir, 'model_best_loss')
                torch.save(model, model_path)
            if val_acc > best_vacc:
                print(f'Best vacc model saving: {best_vacc}-->{val_acc}')
                best_vacc = val_acc
                model_path = os.path.join(out_dir, 'model_best_acc')
                torch.save(model, model_path)
    writer.flush()
    writer.close()
    # sys.exit(0)


if __name__ == '__main__':
    parser = ArgumentParser(description='App description')
    parser.add_argument('--gpu', type=int, default=0)
    # model params
    parser.add_argument('--m_type', type=str, default='tf')
    parser.add_argument('--reg', type=str, default='preset')
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--scheduler', type=str, default='step')
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--w', type=int, default=1)
    parser.add_argument('--d', type=int, default=16)
    # training params
    parser.add_argument('--batch_size', type=int, default=192)
    parser.add_argument('--iter', type=int, default=100000)
    parser.add_argument('--eval_freq', type=int, default=1000)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--save_dir', type=str, required=True)
    # attack params
    parser.add_argument('--attack', type=str, default='attacks.RandomCrop')
    parser.add_argument('--seq_length', type=int, default=90)
    parser.add_argument('--attack_batch', type=int, default=64 * 10)
    parser.add_argument('--n_try', type=int)
    parser.add_argument('--loss', type=str, default='xe')
    # ds params
    parser.add_argument('--fname', type=str, required=True, nargs='+')

    FLAGS = parser.parse_args()
    np.random.seed(9)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    main(FLAGS)
