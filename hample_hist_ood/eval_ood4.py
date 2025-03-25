import torch
import torch.utils.data as loader
import numpy as np
import sys
import os
from datetime import datetime
from tqdm import tqdm
from model.Hample import Hample
from sample.script import SSDataset_690
from sample.ood_mod_histone import histone_substitute, random_uniform
from scripts.calc_auroc import calc_rocs


def eval_ood(mod_type, date_time, model_name, TFs):

    # mod_type = sys.argv[1]
    # # ood_training_type = sys.argv[2]
    # date_time = sys.argv[2]
    # model_name = sys.argv[3]
    # TFs = [sys.argv[4]]

    startTime = datetime.now();
    print(datetime.now().strftime("%m-%d %H:%M:%S"))

    # Dataset options

    cell_types = {0: "GM12878", 1: "H1", 2: "HeLa-S3", 3: "HepG2", 4: "K562"}
    for tfactor in TFs:

        # modded_modal = "hist 256"  # Modified Modality  hist  id
        # mod_type = "sub_rnd"  # one_value_1000 sub_rnd difftf  id

        # --MODIFICATIONS--
        # set histone features of one class to values from other classes randomly
        if mod_type == "sub_rnd":
            histone_substitute(tfactor=tfactor, )
        elif mod_type == "random_uniform":
            random_uniform(tf=tfactor)

        model = Hample()

        # self.model.load_state_dict(torch.load(path + '\\' + self.TF + '.pth', map_location='cpu'))
        model.load_state_dict(
            torch.load("./" + "/".join(["outmodels", tfactor, model_name.split("_")[0], date_time, model_name]) + '.pth',
                       map_location='cpu'))
        model.eval()

        # Predictions
        predicted_values = []
        # todo O: add modified histone inside loader instead of loading from an external file
        TestLoader_ood = loader.DataLoader(dataset=SSDataset_690(tfactor + "_mod" if mod_type != "id" else tfactor,3, True, ), batch_size=128, shuffle=False, num_workers=0)

        for sequence, shape, epigenome, labels in tqdm(TestLoader_ood):
            # for sequence, shape, epigenome, labels in TestLoader_ood:
            # bs=1 (default)
            # cell_num, bs, 1
            binding_predictions = model(sequence.float(), shape.float(), epigenome.float())

            # for prediction in binding_predictions:
            #     """ To scalar"""
            predicted_values.append(binding_predictions.detach().to("cpu").numpy())

        predicted_values_arr = np.empty((0, 5))
        for prediction in predicted_values:
            prediction = prediction.squeeze().T
            predicted_values_arr = np.append(predicted_values_arr, prediction, axis=0)

        print('---Finish Inference on out-of-distribution examples!---\n', datetime.now())
        # predicted_values_arr = np.array(predicted_values)

        # batch size 1
        # predicted_values = []
        # ground_labels = []
        # TestLoader = loader.DataLoader(dataset=SSDataset_690(TF_mod, 3, True), batch_size=1, shuffle=False, num_workers=0)
        # for sequence, shape, epigenome, labels in TestLoader:
        #     binding_predictions = model(sequence.float(), shape.float(), epigenome.float())
        #     labels = labels.permute(1, 0)
        #     for prediction, label in zip(binding_predictions, labels):
        #         predicted_values.append(prediction.squeeze(dim=0).squeeze(dim=0).detach().numpy())
        #         ground_labels.append(label.squeeze(dim=0).detach().numpy())
        # cells_ground_labels_id_orig = np.array(ground_labels).reshape(int(len(ground_labels) / 5), 5)
        # cells_predicted_values_id_orig = np.array(predicted_values).reshape(int(len(ground_labels) / 5), 5)

        cells_predicted_values = predicted_values_arr
        # cells_predicted_values = np.array(predicted_values_arr).reshape(int(len(predicted_values_arr)), 5)
        # original_seq.loc[:,3].value_counts()  label distribution

        np.save("./" + "/".join(
            ["outmodels", tfactor, model_name.split("_")[0], date_time, mod_type]) + "_predicted_values.npy",
                cells_predicted_values)

        print("Using model {} and TF {} and training type {}:".format(model_name, tfactor, mod_type))
        print('---Finished and Saved Predictions!---\t', datetime.now(), datetime.now() - startTime)

        auroc_out = calc_rocs(tfactor, mod_type, model_name=model_name, date_time=date_time)
        os.system("echo " + str(auroc_out) + " >> " + date_time + mod_type + ".txt")

if __name__ == "__main__":
   mod_type = sys.argv[1]
   date_time = sys.argv[2]
   model_name = sys.argv[3]
   TFs = [sys.argv[4]]
   eval_ood(mod_type, date_time, model_name, TFs)

## example settings
# mod_type = "sub_rnd"
# date_time = "03_22_21_37_14"
# model_name = "id"
# tfactor = "GABPA"
## $> python3 eval_ood4.py id 07_15_16_59_38 randuni_05 $TF