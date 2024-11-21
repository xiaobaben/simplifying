import torch
import os
from trainer.pretrain import Pretrain
from trainer.simplify import simplify
from utils.utils import fix_randomness
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from dataloader.dataloader import Build_loader


def main():
    parser = dict()
    parser["epoch"] = 50
    parser["batch"] = 32
    parser["alpha"] = 0.9
    parser["gpu"] = 6
    parser["KFold"] = 5
    parser["rand"] = 1024
    parser["lr"] = 0.0001
    parser['d_lr'] = 0.00005
    parser["f_lr"] = 0.00005
    parser["filepath"] = f"/"
    parser['saved_path'] = f''
    parser["optimizer"] = "AdamW"
    parser["device"] = torch.device(f"cuda:{parser['gpu']}" if torch.cuda.is_available() else "cpu")
    parser["classifier_1"] = 0.5
    parser["adversarial_w"] = 0.3
    parser["classifier_2"] = 0.7
    parser["discrepancy"] = 0.5
    parser["discrepancy_classifier"] = 0.5
    parser["beta"] = [0.5, 0.99]
    parser["weight_decay"] = 3e-4
    parser["num_worker"] = 4
    parser["n_critic"] = 1
    for p in parser.keys():
        print(f"{p}:    {parser[p]}")
    parser['pretrain'] = False
    parser["print_p"] = True
    parser["cross_acc"] = []
    parser["cross_f1"] = []
    parser["cross_mtx"] = []
    parser["cross_dev_loss"] = []
    parser["Fold"] = 1

    fix_randomness(parser["rand"])
    kf = KFold(n_splits=parser["KFold"], shuffle=True, random_state=42)
    torch.multiprocessing.set_start_method('spawn')
    path = [i for i in range(1, 101) if i not in [8, 40]]  # ISRUC
    path_name = {int(j): [[], []] for j in path}
    for t_idx in path:
        num = 0
        file_path = parser['filepath'] + f"/{t_idx}/data"
        label_path = parser['filepath'] + f"/{t_idx}/label"
        while os.path.exists(file_path + f"/{num}.npy"):
            path_name[t_idx][0].append(file_path + f"/{num}.npy")
            path_name[t_idx][1].append(label_path + f"/{num}.npy")
            num += 1

    for fold, (train_idx, val_idx) in enumerate(kf.split(path)):
        print(val_idx)
        train_path = [[], []]
        val_path = [[], []]
        for t_idx in train_idx:
            train_path[0].extend(path_name[path[t_idx]][0])
            train_path[1].extend(path_name[path[t_idx]][1])

        for v_idx in val_idx:
            val_path[0].extend(path_name[path[v_idx]][0])
            val_path[1].extend(path_name[path[v_idx]][1])


        train_dataset = Build_loader(train_path)
        dev_dataset = Build_loader(val_path)

        # 加载数据集
        train_loader = DataLoader(dataset=train_dataset, batch_size=parser['batch'],
                                  shuffle=True, num_workers=parser["num_worker"])
        dev_loader = DataLoader(dataset=dev_dataset, batch_size=parser['batch'],
                                shuffle=True, num_workers=parser["num_worker"])

        if parser['pretrain']:
            Pretrain(train_loader, dev_loader, parser)
        else:
            simplify(train_loader, dev_loader, parser)

        parser['Fold'] += 1

    mtx = None
    for m in range(len(parser["cross_mtx"])):
        if m == 0:
            mtx = parser["cross_mtx"][0]
        else:
            mtx += parser["cross_mtx"][m]

    print("**********Cross   ACC*******************")
    print(parser["cross_acc"])
    print("**********Cross   F1********************")
    print(parser["cross_f1"])

    print("Best Sleep Result")
    print(f"Mean Acc:", sum(parser["cross_acc"])/len(parser["cross_acc"]),
          "Mean Macro F1:", sum(parser["cross_f1"])/len(parser["cross_f1"]))


if __name__ == '__main__':
    main()