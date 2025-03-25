import numpy as np
from sklearn.metrics import roc_auc_score
from Utils.Embedding import LabelEmbedding

"""
Calculate (OOD or ID) auROC scores
"""


def calc_rocs(tfactor, mod_type, model_name, date_time):

    # experimental settings
    # TFs = ["EZH2", "GABPA", "JUND", "MAX", "NRF1", "RFX5", "TAF1",]
    cell_types = {0: "GM12878", 1: "H1", 2: "HeLa-S3", 3: "HpeG2", 4: "K562"}

    # # check for labels and load them; create labels data file if it does not exist
    # labels_path_str = r"./sample/" + tfactor + "/" + tfactor.lower() + "_HCSM_ground_labels_id_orig.npy"
    # if os.path.exists(labels_path_str):
    #     id_labels = np.load("./sample/" + tfactor + "/" + tfactor.lower() + "_HCSM_ground_labels_id_orig.npy")
    # else:
    original_seq = np.genfromtxt(r"./sample/" + tfactor + "/sequence/Test.csv", delimiter=',')
    seq_num = original_seq.shape[0]
    id_labels = np.empty(shape=(seq_num, 5))  # cell_num = 5
    for i in range(seq_num):
        id_labels[i] = LabelEmbedding(original_label=int(original_seq[i, 3]), cell_num=5)

    # in-distribution performance score
    if mod_type == "id":
        preds = np.load("./outmodels/" + tfactor + "/" + "/".join(
            [model_name.split("_")[0], date_time, mod_type]) + "_predicted_values.npy")
        id_auroc = roc_auc_score(id_labels.flatten(), preds.flatten())
        print("Flatten ID AUROC: ", round(id_auroc,4),)
        out_auroc = id_auroc

        class_auroc_list = []
        for i in range(5):
            id_preds = np.load("./" + "/".join(["outmodels", tfactor, model_name.split("_")[0], date_time,
                                                 "id"]) + "_predicted_values.npy")
            id_class_auroc = roc_auc_score(id_labels[:, i], id_preds[:, i])
            print(id_class_auroc, )
            class_auroc_list.append(id_class_auroc)
        print("Mean: ", np.mean(class_auroc_list),"\n", *class_auroc_list)

    # auroc for ood modification types
    else:
        out_auroc = []
        for i in range(5):

            id_preds = np.load("./" + "/".join(["outmodels", tfactor, model_name.split("_")[0], date_time, "id"]) + "_predicted_values.npy")  # _HCSM_predicted_values_id_orig.npy
            ood_preds = np.load("./" + "/".join(["outmodels", tfactor, model_name.split("_")[0], date_time, mod_type]) + "_predicted_values.npy")  # _HCSM_predicted_values_id_orig.npy

            mask_pos = id_labels[:, i] == 1
            id_preds, ood_preds = id_preds[mask_pos], ood_preds[mask_pos]
            ood_labels = np.concatenate([np.zeros(id_preds.shape[0]), np.ones(id_preds.shape[0])])

            ood_auroc = roc_auc_score(ood_labels, np.concatenate([ood_preds[:,i], id_preds[:,i]]))
            print(ood_auroc, )
            out_auroc.append(ood_auroc)
        print(*out_auroc, "\n", round(np.mean(out_auroc),4))
    return np.mean(out_auroc)


if __name__ == "__main__":
    tfactor, mod_type = "GABPA", "id"
    modelname, rundatetime = "mixin", "09_02_15_49_54"
    print("id")
    calc_rocs(tfactor=tfactor, mod_type="id", model_name=modelname, date_time=rundatetime)
    print("random uniform")
    calc_rocs(tfactor=tfactor, mod_type="random_uniform", model_name=modelname, date_time=rundatetime)
    print("sub random")
    calc_rocs(tfactor=tfactor, mod_type="sub_rnd", model_name=modelname, date_time=rundatetime)
    #                   mod_types: id, random_uniform, sub_rnd     model_names: mixin batchswitch randuni
