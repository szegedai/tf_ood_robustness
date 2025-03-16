import collections
import glob
import sys
from argparse import ArgumentParser
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from datareader import ZengData
import os
import pandas as pd
import util
import time
import metrics
from collections import OrderedDict


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(params):
    print(params)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tls = collections.OrderedDict()
    model = torch.load(params.model_path)
    model.eval()
    if len(params.data_path) == 1:
        params.data_path = glob.glob(params.data_path[0])
    for dpi in params.data_path:
        dpi = dpi.replace('\\', '/')
        ds = ZengData(f_path=dpi, binary_label=True, flip_chanels=True, ds_set=params.set)
        # ds = DataLoader(params.data_path, ds_set=params.set)
        x, y = ds.get_data()
        test_loader = torch.utils.data.DataLoader(TensorDataset(torch.Tensor(x), torch.Tensor(y)),
                                                  batch_size=params.batch_size, shuffle=False)
        # print(x.shape, y.shape)
        tli = tls.get(ds.get_name(), [])
        tli.append((test_loader, dpi, ds))
        tls[ds.get_name()] = tli
    dicts = []
    for oidx, k in enumerate(tls.keys()):
        for tli, dpi, ds in tls[k]:
            print(tli, dpi)
            x, y = ds.get_data()
            preds_test = np.zeros_like(y, dtype=np.float32)
            i = 0
            n_correct = 0
            with torch.no_grad():
                for data in tli:
                    inputs, labels = data[0].to(device), data[1].to(device)
                    # attack = make_attack(params.attack, model, params)
                    predi = model(inputs)[:, oidx:(oidx + 1)]
                    n_correcti = torch.sum((predi >= 0 + 0) == labels)
                    n_correct += n_correcti
                    preds_test[i:(i + inputs.size()[0])] = predi.cpu()
                    i += predi.size()[0]

            # preds_test = np.vstack(preds_test)
            # preds_test = model.predict(attack(x, y)[0], batch_size=params.batch_size)
            # plt.hist(preds_test)
            # plt.show()
            # sys.exit(0)
            d = OrderedDict([
                ("ds", dpi),
                ("tf", None if ds.is_deep_sea() else dpi.split("/")[-2]),
                ("mode", None if ds.is_deep_sea() else dpi.split("/")[-3]),
                ("m_selection", params.model_path.split("/")[-1]),
                ("params", count_parameters(model)),
                # ("attack", attack.get_name()),
                ("seq_length", params.seq_length),
                ("train_attack",
                 params.model_path.split("/")[-2].split('_')[-1] if ds.is_deep_sea() else
                 params.model_path.split("/")[-2]),
                ("train_mode",
                 params.model_path.split("/")[-2].split('_')[-3] if ds.is_deep_sea() else
                 params.model_path.split("/")[-4]),
                ("path", params.model_path),
                ("time", time.asctime(time.localtime(time.time())))
            ])
            single = True
            for idx, p in enumerate(params.model_path.split("/")):
                d["path-" + str(idx).zfill(2)] = p
                if p == 'all':
                    single = False
            d['single-tf'] = single
            d['Dist'] = "IN" if d["train_mode"] == d["mode"] else "OUT"

            for m in params.metric.split(','):
                if m == 'bacc':
                    corr_preds_test = ((preds_test >= 0 + 0) == y)
                    score = np.sum(corr_preds_test) / x.shape[0]
                elif m == 'auc':
                    score = metrics.avg_auroc(y, preds_test)
                elif m == 'aupr':
                    score = metrics.avg_auprc(y, preds_test)
                elif m == 'xe':
                    score = np.mean(np.log(np.maximum(np.sum(y * preds_test, axis=1), 1e-4)))
                elif m == 'bce':
                    eps = 1e-4
                    score = np.mean(np.mean(np.where(y,
                                                     -np.log(np.maximum(preds_test, eps)),
                                                     -np.log(np.maximum(1 - preds_test, eps))),
                                            axis=1))
                else:
                    raise Exception('Unknown metric: {0}'.format(m))
                d['adv_' + m] = score
                print('adv_' + m, score)
            dicts.append(d)
            print(dicts[-1])

    if params.out_fname is not None:
        df = pd.DataFrame(data=dicts)
        out_csv_name = params.out_fname
        util.mk_parent_dir(out_csv_name)
        df.to_csv(out_csv_name, index=False, mode='a', header=not os.path.exists(out_csv_name))


if __name__ == '__main__':
    parser = ArgumentParser(description='App description')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=1)
    # attack parameters
    parser.add_argument('--batch_size', type=int, default=4096)  # 3376
    parser.add_argument('--attack', type=str, default='attacks.RandomCrop')
    parser.add_argument('--seq_length', type=int, required=True)
    parser.add_argument('--attack_batch', type=int, default=64)
    parser.add_argument('--loss', type=str, required=True)
    parser.add_argument('--n_try', type=int)
    # dataset paremeters
    parser.add_argument('--set', type=str, default='val')
    parser.add_argument('--data_path', type=str, required=True, nargs='+')
    # loaded model parameters
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--metric', type=str, required=True)
    # output params
    parser.add_argument('--out_fname', type=str)

    FLAGS = parser.parse_args()
    np.random.seed(9)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

    main(FLAGS)
