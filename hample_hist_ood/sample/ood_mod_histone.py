import pandas as pd
import numpy as np
import os

"""
Create OOD datasets for evaluations.
"""
def histone_substitute(tfactor: str, ml_set ="Test.csv"):

    histone = pd.read_csv(os.path.abspath(os.path.dirname(os.path.realpath(__file__))) +"/" + tfactor + "/HM_101/" + ml_set, header=None, index_col=None)
    seq_labels = pd.read_csv(os.path.abspath(os.path.dirname(os.path.realpath(__file__))) +"/" + tfactor + "/sequence/" + ml_set, header=None, index_col=None)

    num = histone.shape[0] // 8
    histone = histone.values
    histone = np.array(np.split(histone, num))
    histone_mod = histone.copy()

    for class_to_change_idx in range(5):  # class_to_change_idx: int,
        # get N_class0 amount of random indexes for other class
        other_class_indexes = np.where(seq_labels[3] != class_to_change_idx)[0]
        substitute_indexes = np.random.choice(other_class_indexes, size=len(seq_labels) - len(other_class_indexes))
        histone_mod[np.where(seq_labels[3] == class_to_change_idx)[0]] = histone[substitute_indexes]

    if not os.path.exists(os.path.abspath(os.path.dirname(os.path.realpath(__file__))) +"/" + tfactor + "_mod/HM_101/"):
        os.popen(" ".join(["cp", "-r", "sample/"+tfactor, "sample/"+tfactor+"_mod"])).read()
    np.savetxt(os.path.abspath(os.path.dirname(os.path.realpath(__file__))) +"/" + tfactor + "_mod/HM_101/" + ml_set, histone_mod.reshape(-1, 101), delimiter=",")


def random_uniform(tfactor: str, ml_set="Test.csv"):
    if not os.path.exists(os.path.abspath(os.path.dirname(os.path.realpath(__file__))) +"/" + tfactor + "_mod/"):
        os.popen(" ".join(["cp", "-r", "sample/"+tfactor, "sample/"+tfactor+"_mod"]))
    histone = pd.read_csv("./sample/"+tfactor + "/HM_101/" + ml_set, header=None, index_col=None)
    fill_values = np.random.normal(7.5,15, size=histone.values.shape)
    # fill_values = np.random.rand(*histone.values.shape)
    np.savetxt("./sample/"+tfactor + "/HM_101/" + ml_set, fill_values, delimiter=",", fmt="%10.3f")
    print("Done converting the following TFs: ", tfactor+" "+ml_set, "with modification: ", random_uniform.__name__)

