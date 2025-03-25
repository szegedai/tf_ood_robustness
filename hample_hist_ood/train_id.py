import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch.nn as nn
import torch.optim as optim
import math
from datetime import datetime

from scripts.ood_train_mods import *
import eval_ood4
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc, f1_score
from torch.utils.data import random_split
from sample.script import SSDataset_690
from Utils.EarlyStopping import EarlyStopping
from model.Hample import Hample


class Trainer:

    def __init__(self, model, model_name, TF,
                 batch_size, epochs, cell_num):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device=self.device)
        self.model_name = model_name[0]
        self.model_suffix = model_name[1]
        self.TF = TF
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, patience=3, factor=0.5)
        self.loss_function = nn.BCELoss()

        self.batch_size = batch_size
        self.epochs = epochs
        self.cell_num = cell_num

        self.sequence_order = 3  # default(3) empiric
        self.date_time = datetime.now().strftime("%m_%d_%H_%M_%S")
        self.exp_desc = model_name[2]

    def learn(self, TrainLoader, ValidateLoader):

        path = os.path.abspath(os.curdir) + "/" + "/".join(["outmodels", self.TF, self.model_name, self.date_time])
        print(self.date_time)
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True)
        history = np.empty((self.epochs, 6))

        for epoch in range(self.epochs):
            self.model.to(self.device)
            self.model.train()

            train_id_loss_list, train_ood_loss_list = [], []

            ProgressBar = tqdm(TrainLoader)
            for data in ProgressBar:
                self.optimizer.zero_grad()

                ProgressBar.set_description("Epoch %d" % epoch)
                sequence, shape, epigenome, labels = data

                final_loss_ood = torch.Tensor([0])

                binding_predictions = self.model(sequence.to(self.device, dtype=torch.float),
                                                 shape.to(self.device, dtype=torch.float),
                                                 epigenome.to(self.device, dtype=torch.float))

                # cell_num, bs
                labels = labels.permute(1, 0)
                final_loss = 0
                for prediction_in_bs, label_in_bs in zip(binding_predictions, labels):
                    final_loss = final_loss + self.loss_function(prediction_in_bs,
                                                                 label_in_bs.float().unsqueeze(1).to(self.device))

                final_loss = final_loss / self.cell_num

                train_id_loss_list.append(final_loss.item()); train_ood_loss_list.append(final_loss_ood.item())

                # print("Train: ", final_loss, final_loss_ood, (final_loss + final_loss_ood) / 2)
                # average losses (from the id and ood batches)
                # final_loss2, final_loss_ood2 = final_loss.detach().cpu(), final_loss_ood.detach().cpu()
                # final_loss = (final_loss + final_loss_ood) / 2
                # train_loss_list.append((final_loss2, final_loss_ood2, (final_loss2 + final_loss_ood2) / 2))
                # ProgressBar.set_postfix(loss=str(final_loss.item())+" Train: "+" ".join([str(final_loss2.numpy()), str(final_loss_ood2.numpy()), str((final_loss2 + final_loss_ood2) / 2)]))   # módosítva!

                ProgressBar.set_postfix(loss=final_loss.item())

                final_loss.backward()
                self.optimizer.step()

            history[epoch, :3] = [np.mean(train_id_loss_list), np.mean(train_ood_loss_list), (np.mean(train_id_loss_list) + np.mean(train_ood_loss_list)/2)]

            final_valid_loss, final_valid_loss_ood = [], []

            self.model.eval()
            with torch.no_grad():
                for valid_sequence, valid_shape, valid_epigenome, valid_labels in ValidateLoader:
                    # cell_num, bs, 1
                    valid_binding_predictions = self.model(valid_sequence.to(self.device, dtype=torch.float),
                                                           valid_shape.to(self.device, dtype=torch.float),
                                                           valid_epigenome.to(self.device, dtype=torch.float))
                    # cell_num, bs
                    valid_labels = valid_labels.float().to(self.device)
                    valid_labels = valid_labels.permute(1, 0)

                    valid_loss_in_bs = 0
                    for valid_prediction_in_bs, valid_label_in_bs in zip(valid_binding_predictions, valid_labels):
                        valid_loss_in_bs = valid_loss_in_bs + self.loss_function(valid_prediction_in_bs,
                                                                                 valid_label_in_bs.unsqueeze(1)).item()

                    final_valid_loss.append(valid_loss_in_bs / self.cell_num)


                valid_loss_avg = torch.mean(torch.Tensor(final_valid_loss))
                valid_loss_ood_avg = torch.mean(torch.Tensor(final_valid_loss_ood))
                self.scheduler.step(valid_loss_avg)
                print("Val: ", valid_loss_avg.item(), valid_loss_ood_avg.item(), ((valid_loss_avg + valid_loss_ood_avg) / 2).item())

                history[epoch, 3:] = valid_loss_avg.item(), valid_loss_ood_avg.item(), ((valid_loss_avg + valid_loss_ood_avg) / 2).item()


            early_stopping(valid_loss_avg, self.model,
                           path + '/' + self.model_name + self.model_suffix +'.pth')
            torch.save(self.model.state_dict(), path + '/' + self.model_name + self.model_suffix + "_" + str(epoch) + '.pth')
            np.savetxt(path + '/history.csv', history, delimiter=",", fmt="%.5f", header="train_id_loss, train_ood_loss_list, final_train_loss, valid_loss_avg, valid_loss_ood_avg, final_val_loss")


        print('---Finish Learn---')

    def inference(self, TestLoader):

        path = os.path.abspath(os.curdir) + "/" + "/".join(["outmodels", self.TF, self.model_name, self.date_time])


        self.model.load_state_dict(torch.load(path + '/' + self.model_name + self.model_suffix +'.pth', map_location='cpu'))
        self.model.to("cpu")

        predicted_values = []
        ground_labels = []
        self.model.eval()

        for sequence, shape, epigenome, labels in TestLoader:
            # bs=1 (default)
            # cell_num, bs, 1
            binding_predictions = self.model(sequence.float(), shape.float(), epigenome.float())
            # cell_num, bs
            labels = labels.permute(1, 0)
            for prediction, label in zip(binding_predictions, labels):
                """ To scalar"""
                predicted_values.append(prediction.squeeze(dim=0).squeeze(dim=0).detach().numpy())
                ground_labels.append(label.squeeze(dim=0).detach().numpy())

        print('---Finish Inference---')

        return predicted_values, ground_labels

    def measure(self, predicted_values, ground_labels):
        accuracy = accuracy_score(y_pred=np.array(predicted_values).round(), y_true=ground_labels)
        roc_auc = roc_auc_score(y_score=predicted_values, y_true=ground_labels)

        precision, recall, _ = precision_recall_curve(y_score=predicted_values, y_true=ground_labels)
        pr_auc = auc(recall, precision)

        f_score = f1_score(y_pred=np.array(predicted_values).round(), y_true=ground_labels)

        print('\n---Finish Measure---\n')

        return accuracy, roc_auc, pr_auc, f_score

    def save_evaluation_indicators(self, indicators):
        path = os.path.abspath(os.curdir) + "/" + "/".join(["outmodels", self.TF, self.model_name, self.date_time]) + "/"

        if not os.path.exists(path):
            os.makedirs(path)
        #     写入评价指标
        file_name = path + self.model_name + self.model_suffix + "_Indicators.csv"
        file = open(file_name, "a")
        file.write(str(indicators[0]) + " " + str(np.round(indicators[1], 4)) + " " +
                   str(np.round(indicators[2], 4)) + " " + str(np.round(indicators[3], 4)) + " " +
                   str(np.round(indicators[4], 4)) + "\n" + " " + self.exp_desc)

        file.close()

    def run(self, samples_file_name, ratio=0.8):
        """
        Train_Validate_Set = SSDataset_690(samples_file_name, self.sequence_order, False)
        """
        Train_Validate_Set = SSDataset_690(samples_file_name, self.sequence_order, False)

        Train_Set, Validate_Set = random_split(dataset=Train_Validate_Set,
                                               lengths=[math.ceil(len(Train_Validate_Set) * ratio),
                                                        len(Train_Validate_Set) -
                                                        math.ceil(len(Train_Validate_Set) * ratio)],
                                               generator=torch.Generator().manual_seed(0))
        TrainLoader = loader.DataLoader(dataset=Train_Set, drop_last=True,
                                        batch_size=self.batch_size, shuffle=True, num_workers=0)
        ValidateLoader = loader.DataLoader(dataset=Validate_Set, drop_last=True,
                                           batch_size=self.batch_size, shuffle=False, num_workers=0)

        TestLoader = loader.DataLoader(dataset=SSDataset_690(samples_file_name, self.sequence_order, True),
                                       batch_size=1, shuffle=False, num_workers=0)

        self.learn(TrainLoader, ValidateLoader)

        predicted_values, ground_labels = self.inference(TestLoader)

        accuracy, roc_auc, pr_auc, f_score = self.measure(predicted_values, ground_labels)

        # 写入评价指标
        indicators = [self.TF, accuracy, roc_auc, pr_auc, f_score]
        self.save_evaluation_indicators(indicators)

        print('---Finish Run---\n')

        eval_ood4.eval_ood("id", self.date_time, "id", [self.TF])
        eval_ood4.eval_ood("random_uniform", self.date_time, "id", [self.TF])
        eval_ood4.eval_ood("sub_rnd", self.date_time, "id", [self.TF])
        print('---Finished OOD evaluations!---')

def main():

    model_suffix = ""  # sys.argv[1]

    # TFs = ["EZH2", "GABPA", "JUND", "MAX", "NRF1", "RFX5", "TAF1", ]  # USF2 model missing
    TFs = ["GABPA"]  # attacks:  ooddifftf oodall4
    exp_name = ['id', model_suffix, "vanilla",]  # exp_name: model neve, batch increase size, exp_desc

    for TF in TFs:
            Train = Trainer(model=Hample(ham_i_dim=(4,64)), TF=TF, model_name=exp_name, batch_size=64, epochs=1, cell_num=5)
            Train.run(samples_file_name=TF)


main()


"""
Train = Trainer(model=Hample(),
                TF='ATF2', model_name='Hample', batch_size=1, epochs=15, cell_num=5)
Train.run(samples_file_name='USF2')
"""