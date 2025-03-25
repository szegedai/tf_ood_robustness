import numpy as np
import torch
import torch.utils.data as loader
from sklearn.metrics import roc_auc_score

from sample.script import SSDataset_690
import random


def predict_and_loss_on_oods(self, sequence, shape, epigenome_ood, increase_size, return_preds=False):

    if float(increase_size) < 1:
        sequence = torch.clone(sequence[:int(float(increase_size)*self.batch_size)])
        shape = torch.clone(shape[:int(float(increase_size)*self.batch_size)])
    else:
        sequence = sequence.repeat_interleave(int(increase_size), dim=0)
        shape = shape.repeat_interleave(int(increase_size), dim=0)

    binding_predictions = self.model(sequence.to(self.device, dtype=torch.float),
                                     shape.to(self.device, dtype=torch.float),
                                     epigenome_ood.to(self.device, dtype=torch.float))
        # todo O: cpu-n is?
    # debug check input_val 0
    if sum(sum(torch.isnan(binding_predictions))) > 0:
        binding_predictions = torch.nan_to_num(binding_predictions)
        print("error nan 0")

    # labels = torch.full((len(sequence), 5), 0,)
    # labels = labels.permute(1, 0)
    # labels = labels.to(self.device)  # , dtype=torch.float
    # final_loss = self.loss_function(binding_predictions, torch.tensor(np.full((5, int(64*increase_size), 1), 0)).float().to(self.device))
    final_loss = 0
    labels_ood = torch.tensor(np.full((5, int(self.batch_size*float(increase_size)),), 0))
    for prediction_in_bs, label_in_bs in zip(binding_predictions, labels_ood):
        final_loss = final_loss + self.loss_function(prediction_in_bs, label_in_bs.float().unsqueeze(1).to(self.device))
    # final_loss = final_loss / 64
    # final_loss = final_loss / self.cell_num
    final_loss = final_loss / self.cell_num

    if return_preds:
        return final_loss, binding_predictions
    else:
        return final_loss

def calc_ood_auroc(id_labels, id_preds,ood_preds):
    if ood_preds.shape < id_preds.shape:
        cut_idx = np.random.choice(np.arange(id_preds.shape[0]), ood_preds.shape[0], replace=False)
        id_preds = id_preds[cut_idx]
        id_labels = id_labels[cut_idx]
    elif id_preds.shape < ood_preds.shape:
        cut_idx = np.random.choice(np.arange(ood_preds.shape[0]), id_preds.shape[0], replace=False)
        ood_preds = ood_preds[cut_idx]

    out_auroc = []
    for i in range(5):
        mask_pos = id_labels[:, i] == 1
        id_preds_i, ood_preds_i = id_preds[mask_pos].copy(), ood_preds[mask_pos].copy()
        ood_labels = np.concatenate([np.zeros(id_preds_i.shape[0]), np.ones(id_preds_i.shape[0])])
        ood_auroc = roc_auc_score(ood_labels, np.concatenate([ood_preds_i[:, i], id_preds_i[:, i]]))
        out_auroc.append(ood_auroc)
        # print(np.mean(out_auroc), *out_auroc)
    return np.mean(out_auroc)


def randuni(increase_size, batch_size=64):
    ood_epigenome = np.random.normal(7.5,15, size = (int(batch_size * increase_size), 8, 101),)

    # ood_epigenome = np.random.random((int(64 * increase_size), 8, 101),)
    return torch.from_numpy(ood_epigenome)


def change_histone_in_batch(epigenome, labels, increase_size):  # mixin
    change_amount_in_batch = 64*float(increase_size)  # 16/64 = .25
    epigenome_ood = []
    to_be_changed_idx = np.random.choice(np.arange(64), int(change_amount_in_batch), replace=False)
    for i in to_be_changed_idx:
        other_class_idx = np.where(1 != labels[:, np.argmax(labels[i])])
        if len(other_class_idx[0]) < 1:
            continue
        adv_hist = epigenome[np.random.choice(other_class_idx[0])]
        epigenome_ood.append(adv_hist)  # todo O: futási idő ellenőrzése
        # labels[i] = torch.from_numpy(np.array([0, 0, 0, 0, 0]))  # todo O: futási idő ellenőrzése
    return torch.from_numpy(np.array(epigenome_ood))


def create_all_oods(TrainLoader, epigenome, labels, increase_size):  # batchswitch
    ood_epigenome = torch.repeat_interleave(epigenome, int(increase_size), dim=0)
    # orig01:    ood_epigenome = torch.empty(4*epigenome.shape[0], epigenome.shape[1], epigenome.shape[2])
    labels = labels.repeat_interleave(int(increase_size), dim=0)  # O: todo fix enumerate and labels - elég 64?

    for i in range(0, len(labels), 4):

        for j, j_idx in enumerate(np.delete(np.arange(5), np.argmax(labels[i]))):

            # select histone from other classes
            class_hist_idx = np.where(1 == TrainLoader.dataset.dataset.completed_labels[:, j_idx])[0]

            adv_hist = TrainLoader.dataset.dataset.completed_histone[np.random.choice(class_hist_idx)]
            ood_epigenome[i+j] = torch.from_numpy(adv_hist)
    if sum(sum(sum(torch.isnan(ood_epigenome)))) > 0:
        ood_epigenome = torch.nan_to_num(ood_epigenome)
        print("error nan ood_epigenome")
    return ood_epigenome



def loaders_from_other_tfs(tf_names=["EZH2", ]):  # GABPA "JUND", "MAX", "NRF1"
    loader_dict = {}  # todo O: összes hisztont kigyűjteni egy nagy fáljba és abból generálni (osztályokra figyelve\nem)
    for tf_name in tf_names:
        data_loader = loader.DataLoader(dataset=SSDataset_690(tf_name+"", 3, False),
                                        batch_size=64, shuffle=True, num_workers=0)
        loader_dict[tf_name] = data_loader
        print("Data Loader created for: ", tf_name)
    return loader_dict


def change_histone_in_batch_from_difftfloader(other_TFs_loader_dict, labels, increase_size):
    change_amount_in_batch = 8  # 16/64 = .25   # batch_multiplier = 2  -> change_amount_in_batch = (64 / 4) * batch_multiplier
    ood_epigenome = np.ones((int(64 * increase_size), 8, 101), dtype="float32")

    for i, label in enumerate(labels):
        for j in range(int(increase_size)):
            curr_loader = random.choice(list(other_TFs_loader_dict.values()))
            iter_curr_loader = iter(curr_loader)
            while True:
                next_ood_batch = next(iter_curr_loader)
                diff_label_indices = np.where(np.argmax(next_ood_batch[3], axis=1) != np.argmax(labels[i]))[0]
                if len(diff_label_indices) == 0:
                    continue
                else:
                    ood_epigenome[i+j] = next_ood_batch[2][np.random.choice(diff_label_indices)]
                    break
    return torch.from_numpy(ood_epigenome)


# régi change histone in batch from difftfloader - pontos arányokkal dolgozott a többi tf loaderéből
    # def change_histone_in_batch_from_difftfloader(other_TFs_loader_dict, labels):
    #     change_amount_in_batch = 8  # 16/64 = .25   # batch_multiplier = 2  -> change_amount_in_batch = (64 / 4) * batch_multiplier
    #     ood_epigenome = np.empty((64, 8, 101))
    #
    #     for j, curr_loader in enumerate(other_TFs_loader_dict.values()):
    #         iter_curr_loader = iter(curr_loader)
    #         j2 = j * change_amount_in_batch
    #         for i in range(j2, j2 + change_amount_in_batch):
    #             while True:
    #                 next_ood_batch = next(iter_curr_loader)
    #                 diff_label_indices = np.where(np.argmax(next_ood_batch[3], axis=1) != np.argmax(labels[i]))[0]
    #                 if len(diff_label_indices) == 0:
    #                     continue
    #                 else:
    #                     ood_epigenome[i] = next_ood_batch[2][np.random.choice(diff_label_indices)]
    #                     break
    #
    #     return torch.from_numpy(ood_epigenome)

    #
    # to_be_changed_idx = np.random.choice(np.arange(64), change_amount_in_batch, replace=False)
    # for i in to_be_changed_idx:
    #     other_class_idx = np.where(1 != labels[:, np.argmax(labels[i])])
    #     adv_hist = epigenome[np.random.choice(other_class_idx[0])]
    #     epigenome[i] = adv_hist  # todo O: futási idő ellenőrzése
    #     labels[i] = torch.from_numpy(np.array([0, 0, 0, 0, 0]))  # todo O: futási idő ellenőrzése
    # return ood_epigenome

    # dataloader_iterator = iter(dataloader)
    # for i in range(iterations):
    #     try:
    #         data, target = next(dataloader_iterator)
    #     except StopIteration:
    #         dataloader_iterator = iter(dataloader)
    #         data, target = next(dataloader_iterator)
    #     do_something()
