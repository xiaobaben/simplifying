# Time : 2023/2/16 14:29
# Author : 小霸奔
# FileName: EogSimplify.p
import collections

from model.model import FeatureExtractor, SleepMLP, AttFuEncode
import torch.nn as nn
import torch
import numpy as np
from utils.configs import ModelConfig
from sklearn.metrics import classification_report, f1_score, confusion_matrix


def Pretrain(train_dl, val_dl, args):
    """
    :param train_dl: train set dataloader
    :param val_dl: val set dataloader
    :param args: train parameters
    :return:
    """
    # Initialize parameter
    device = args["device"]
    total_acc = []
    total_f1 = []
    total_mtx = []
    total_dev_loss = []
    best_acc = 0
    best_f1 = 0
    best_epoch = 0
    es = 0
    # Build model network
    feature_extractor = FeatureExtractor().float().to(device)
    sleep_classifier = SleepMLP().float().to(device)
    att_encoder = AttFuEncode().float().to(device)

    if args["print_p"]:
        num_para = 0
        for model in [feature_extractor, sleep_classifier, att_encoder]:
            num_para += sum(p.numel() for p in model.parameters())
        print(f"parameter num:    {num_para}")
        args["print_p"] = False
    # loss function
    classifier_criterion = nn.CrossEntropyLoss().to(device)

    # optimizer
    optimizer_encoder = torch.optim.Adam(list(feature_extractor.parameters())
                                         + list(sleep_classifier.parameters())
                                         + list(att_encoder.parameters()), lr=args["lr"],
                                         betas=(args['beta'][0], args['beta'][1]),
                                         weight_decay=args['weight_decay'])
    model_param = ModelConfig()
    train_loss = []
    for epoch in range(1, args["epoch"]+1):
        print(f" KFold:{args['Fold']}-------------epoch{epoch}---Train--------------------------------")
        feature_extractor.train()
        sleep_classifier.train()
        att_encoder.train()
        running_loss = 0.0

        batch_loss = []
        for batch_idx, data in enumerate(train_dl):
            eog, eeg, gaussian_noise, label = data[0].to(device), data[1].to(device), \
                                              data[2].to(device), data[3].to(device)
            batch_size = eog.shape[0]
            length = model_param.SeqLength
            epoch_size = model_param.EpochLength

            eog = eog.view(-1, model_param.EogNum, epoch_size)
            eeg = eeg.view(-1, model_param.EegNum, epoch_size)

            e1 = eog[:, 0, :].view(batch_size * length, 1, epoch_size)
            e2 = eog[:, 1, :].view(batch_size * length, 1, epoch_size)

            f3 = eeg[:, 0, :].view(batch_size * length, 1, epoch_size)
            c3 = eeg[:, 1, :].view(batch_size * length, 1, epoch_size)
            o1 = eeg[:, 2, :].view(batch_size * length, 1, epoch_size)
            f4 = eeg[:, 3, :].view(batch_size * length, 1, epoch_size)
            c4 = eeg[:, 4, :].view(batch_size * length, 1, epoch_size)
            o2 = eeg[:, 5, :].view(batch_size * length, 1, epoch_size)

            e1 = feature_extractor(e1)
            e2 = feature_extractor(e2)

            f3 = feature_extractor(f3)
            c3 = feature_extractor(c3)
            o1 = feature_extractor(o1)
            f4 = feature_extractor(f4)
            c4 = feature_extractor(c4)
            o2 = feature_extractor(o2)

            # EEG + EOG
            eeg_eog_feature = att_encoder(f3, c3, f4, c4, e1, e2)  # batch, 20, 512
            pred = sleep_classifier(eeg_eog_feature)

            # Compute  Classification Loss
            loss_classifier = classifier_criterion(pred, label.long())

            total_loss = loss_classifier
            optimizer_encoder.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(sleep_classifier.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(feature_extractor.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(att_encoder.parameters(), 1.0)
            optimizer_encoder.step()

            running_loss += total_loss.item()
            if batch_idx % 10 == 9:  # 输出每次的平均loss
                print('\n [%d,  %5d] total_loss: %.3f  ' %
                      (epoch, batch_idx + 1, running_loss / 10))
                batch_loss.append(running_loss / 10)
                running_loss = 0.0
        train_loss.append(np.mean(batch_loss))

        if epoch % 1 == 0:
            print(f" KFold:{args['Fold']}-------------epoch{epoch}---Val----------------------------------")
            report = eog_simplify_dev((feature_extractor, att_encoder, sleep_classifier),
                                      val_dl, args, model_param)
            total_acc.append(report[0])
            total_f1.append(report[1])
            total_dev_loss.append(report[2])
            total_mtx.append(report[3])

        if total_acc[-1] > best_acc:
            best_acc = total_acc[-1]
            best_f1 = total_f1[-1]
            es = 0
            best_epoch = epoch
            best_feature = (feature_extractor, att_encoder, sleep_classifier)
        else:
            es += 1
            print(f"{args['Fold']}Fold   Counter {es} of 20")

            if es > 20 and epoch > 50:
                print(f"{args['Fold']}Fold   Early stopping with best_acc: {best_acc}   best f1: {best_f1}")
                # 最好的睡眠分期结果
                args['cross_acc'].append(best_acc)
                args['cross_f1'].append(best_f1)
                args['cross_mtx'].append(total_mtx[best_epoch-1])
                args['cross_dev_loss'].append(total_dev_loss)
                multi_modality_model = best_feature
                state_f = multi_modality_model[0].state_dict()
                for key in state_f.keys():
                    state_f[key] = state_f[key].to(torch.device("cpu"))

                state_att = multi_modality_model[1].state_dict()
                for key in state_att.keys():
                    state_att[key] = state_att[key].to(torch.device("cpu"))

                state_sleep = multi_modality_model[2].state_dict()
                for key in state_sleep.keys():
                    state_sleep[key] = state_sleep[key].to(torch.device("cpu"))

                torch.save(state_f, f"{args['saved_path']}/{args['Fold']}/feature_extractor_parameter.pkl")
                torch.save(state_att, f"{args['saved_path']}/{args['Fold']}/att_encoder_parameter.pkl")
                torch.save(state_sleep, f"{args['saved_path']}/{args['Fold']}/sleep_classifier_parameter.pkl")
                break
        print("dev_acc:", total_acc)
        print("dev_macro_f1:", total_f1)
    else:
        args['cross_acc'].append(best_acc)
        args['cross_f1'].append(best_f1)
        args['cross_mtx'].append(total_mtx[best_epoch - 1])
        args['cross_dev_loss'].append(total_dev_loss)


def eog_simplify_dev(model, val_dl, args, model_param):
    """
    :param model: (feature_extractor, att_encoder, sleep_classifier)
    :param val_dl: Val Set Dataloader
    :param args: Val parameters
    :param model_param: Model Parameters
    :return: report: tuple(acc, macro_f1, dev_mean_loss/count, confusion_mtx)
    """
    if type(model) == tuple:
        model[0].eval()
        model[1].eval()
        model[2].eval()
    else:
        model.eval()

    device = args["device"]
    criterion = nn.CrossEntropyLoss()

    y_pred = []
    y_test = []
    with torch.no_grad():
        correct = 0.0
        total = 0.0
        dev_mean_loss = 0.0
        for batch_idx, data in enumerate(val_dl):
            eog, eeg, gaussian_noise, labels = data[0].to(device), data[1].to(device), \
                                              data[2].to(device), data[3].to(device)
            batch_size = eog.shape[0]
            length = model_param.SeqLength
            epoch_size = model_param.EpochLength

            eog = eog.view(-1, model_param.EogNum, epoch_size)
            eeg = eeg.view(-1, model_param.EegNum, epoch_size)

            e1 = eog[:, 0, :].view(batch_size * length, 1, epoch_size)
            e2 = eog[:, 1, :].view(batch_size * length, 1, epoch_size)

            f3 = eeg[:, 0, :].view(batch_size * length, 1, epoch_size)
            c3 = eeg[:, 1, :].view(batch_size * length, 1, epoch_size)
            o1 = eeg[:, 2, :].view(batch_size * length, 1, epoch_size)
            f4 = eeg[:, 3, :].view(batch_size * length, 1, epoch_size)
            c4 = eeg[:, 4, :].view(batch_size * length, 1, epoch_size)
            o2 = eeg[:, 5, :].view(batch_size * length, 1, epoch_size)

            e1 = model[0](e1)
            e2 = model[0](e2)

            f3 = model[0](f3)
            c3 = model[0](c3)
            o1 = model[0](o1)
            f4 = model[0](f4)
            c4 = model[0](c4)
            o2 = model[0](o2)

            # EEG + EOG
            eeg_eog_feature = model[1](f3, c3, f4, c4, e1, e2)  # batch, 20, 512
            prediction = model[2](eeg_eog_feature)

            dev_loss = criterion(prediction, labels.long())
            dev_mean_loss += dev_loss.item()

            _, predicted = torch.max(prediction.data, dim=1)
            predicted, labels = torch.flatten(predicted), torch.flatten(labels)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            acc = correct / total
            count = batch_idx
            predicted = predicted.tolist()
            y_pred.extend(predicted)
            labels = labels.tolist()
            y_test.extend(labels)

        macro_f1 = f1_score(y_test, y_pred, average="macro")
        print('dev loss:', dev_mean_loss / count, 'Accuracy on sleep:', acc, 'F1 score on sleep:', macro_f1, )
        print(classification_report(y_test, y_pred, target_names=['Sleep stage W',
                                                                  'Sleep stage 1',
                                                                  'Sleep stage 2',
                                                                  'Sleep stage 3/4',
                                                                  'Sleep stage R']))
        confusion_mtx = confusion_matrix(y_test, y_pred)
        print(confusion_mtx)

        report = (acc, macro_f1, dev_mean_loss/count, confusion_mtx)
        return report


