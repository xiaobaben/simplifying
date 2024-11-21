# Time : 2023/2/22 13:07
# Author : 小霸奔
# FileName: EogSimplify3.p
"""
两步训练  第一步训练特征提取器，取训练过程中最好结果的多模态特征作为第二步对抗训练的输入
        第二步训练特征生成器，使用第一步训练好的多模态特征作为输入
"""
from model.model import FeatureExtractor, SleepMLP, SleepMLP2, AttFuEncode
import torch.nn as nn
import torch
from utils.configs import ModelConfig
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from model.model import Discriminator_ISRUC, NoiseEogFusion_ISRUC


def simplify(train_dl, val_dl, args):
    """
    multi_modality_model  (feature_extractor, att_encoder, sleep_classifier)
    Pretrain
    """
    feature_extractor = FeatureExtractor()
    att_encoder = AttFuEncode()
    sleep_classifier = SleepMLP()
    feature_extractor.load_state_dict(torch.load(f""))
    att_encoder.load_state_dict(torch.load(f""))

    multi_modality_model = (feature_extractor, att_encoder, sleep_classifier)
    step2_train(train_dl, val_dl, multi_modality_model, args)


def step2_train(train_dl, val_dl, multi_modality_feature, args):
    # Initialize parameter
    device = args["device"]
    total_acc = []
    total_f1 = []
    best_acc = 0
    best_f1 = 0
    best_epoch = 0
    # Build model network
    feature_extractor_eeg_eog = multi_modality_feature[0].to(device)
    att_encoder = multi_modality_feature[1].to(device)

    feature_discriminator = Discriminator_ISRUC().float().to(device)
    feature_generator1 = NoiseEogFusion_ISRUC().float().to(device)
    feature_generator2 = NoiseEogFusion_ISRUC().float().to(device)
    sleep_classifier1 = SleepMLP().float().to(device)
    sleep_classifier2 = SleepMLP2().float().to(device)

    # loss function
    discriminator_criterion = nn.BCELoss().to(device)
    classifier_criterion = nn.CrossEntropyLoss().to(device)
    # optimizer

    optimizer_generator = torch.optim.AdamW([{"params": list(feature_generator1.parameters())},
                                             {"params": list(feature_generator2.parameters())},
                                            {"params": list(sleep_classifier2.parameters())},
                                             {"params": list(sleep_classifier1.parameters())},

                                             ],
                                            lr=args["lr"], betas=(args['beta'][0], args['beta'][1]),
                                            weight_decay=args['weight_decay'])

    optimizer_discriminator = torch.optim.AdamW(feature_discriminator.parameters(), lr=args["d_lr"],
                                                weight_decay=args['weight_decay'], )

    model_param = ModelConfig()

    if args["print_p"]:
        num_para = 0
        for model in [feature_extractor_eeg_eog, feature_generator1, feature_generator2,
                    sleep_classifier1, feature_discriminator,  sleep_classifier2,
                      att_encoder]:
            num_para += sum(p.numel() for p in model.parameters())
        print(f"parameter num:    {num_para}")
        args["print_p"] = False

    for epoch in range(1, args["epoch"]+1):
        print(f" KFold:{args['Fold']}-------------epoch{epoch}---Adversarial_Train--------------------------------")
        feature_extractor_eeg_eog.train()
        att_encoder.train()

        feature_generator1.train()
        feature_generator2.train()
        feature_discriminator.train()
        sleep_classifier1.train()
        sleep_classifier2.train()


        running_loss = 0.0
        disc_running_loss = 0.0
        for batch_idx, data in enumerate(train_dl):
            eog, eeg, gaussian_noise, label = data[0].to(device), data[1].to(device), \
                                              data[2].to(device), data[3].to(device)
            for param in feature_discriminator.parameters():
                param.requires_grad = True

            batch_size = eog.shape[0]
            length = model_param.SeqLength
            epoch_size = model_param.EpochLength

            eog = eog.view(-1, model_param.EogNum, epoch_size)  # batch*length, 2, 3000
            eeg = eeg.view(-1, model_param.EegNum, epoch_size)

            e11 = eog[:, 0, :].view(batch_size * length, 1, epoch_size)
            e22 = eog[:, 1, :].view(batch_size * length, 1, epoch_size)

            e11_freq = torch.abs(torch.fft.fft(e11))
            e22_freq = torch.abs(torch.fft.fft(e22))

            f3 = eeg[:, 0, :].view(batch_size * length, 1, epoch_size)
            c3 = eeg[:, 1, :].view(batch_size * length, 1, epoch_size)
            o1 = eeg[:, 2, :].view(batch_size * length, 1, epoch_size)
            f4 = eeg[:, 3, :].view(batch_size * length, 1, epoch_size)
            c4 = eeg[:, 4, :].view(batch_size * length, 1, epoch_size)
            o2 = eeg[:, 5, :].view(batch_size * length, 1, epoch_size)

            e1 = feature_extractor_eeg_eog(e11)
            e2 = feature_extractor_eeg_eog(e22)

            f3 = feature_extractor_eeg_eog(f3)
            c3 = feature_extractor_eeg_eog(c3)
            o1 = feature_extractor_eeg_eog(o1)
            f4 = feature_extractor_eeg_eog(f4)
            c4 = feature_extractor_eeg_eog(c4)
            o2 = feature_extractor_eeg_eog(o2)

            # EEG + EOG
            eeg_eog_feature = att_encoder(f4, f3, c4, c3, o1, o2, e1, e2)  # batch, 20, 512
            # E1 E2 concat noise Then Linear
            noise_eog_feature1 = feature_generator1((e11, e22), gaussian_noise)  # 提取时域
            noise_eog_feature2 = feature_generator2((e11_freq, e22_freq), gaussian_noise)  # 提取频域

            noise_eog_feature = args["alpha"]*noise_eog_feature1 + (1-args["alpha"])*noise_eog_feature2
            # noise_eog_feature = fusion(noise_eog_feature1, noise_eog_feature2)

            noise_pred1 = sleep_classifier1(noise_eog_feature)
            noise_pred2 = sleep_classifier2(noise_eog_feature)

            # concatenate True and Fake features
            concat_feat1 = torch.cat((eeg_eog_feature, noise_eog_feature1), dim=0)  # batch*2, 20, 512
            concat_feat2 = torch.cat((eeg_eog_feature, noise_eog_feature2), dim=0)  # batch*2, 20, 512
            concat_e1 = torch.cat((e11, e11), dim=0)
            concat_e2 = torch.cat((e22, e22), dim=0)

            concat_e1_freq = torch.cat((e11_freq, e11_freq), dim=0)
            concat_e2_freq = torch.cat((e22_freq, e22_freq), dim=0)

            # predict the Adversarial label by the discriminator network  # batch*2, 20 ,1
            concat_pred1 = feature_discriminator(concat_feat1.detach(), (concat_e1.detach(), concat_e2.detach()))
            concat_pred2 = feature_discriminator(concat_feat2.detach(), (concat_e1_freq.detach(), concat_e2_freq.detach()))

            # prepare real labels for the training the discriminator
            disc_eeg_eog_labels = torch.ones(size=(eeg_eog_feature.size(0),
                                                   eeg_eog_feature.size(1))).long().to(device)
            disc_eog_noise_label = torch.zeros(size=(noise_eog_feature1.size(0),
                                                     noise_eog_feature1.size(1))).long().to(device)
            label_concat = torch.cat((disc_eeg_eog_labels, disc_eog_noise_label), 0)  # batch*2, 20

            # Discriminator Loss
            loss_disc1 = discriminator_criterion(concat_pred1.squeeze(), label_concat.float())
            loss_disc2 = discriminator_criterion(concat_pred2.squeeze(), label_concat.float())

            loss_disc = 0.5*loss_disc1 + 0.5*loss_disc2
            optimizer_discriminator.zero_grad()
            loss_disc.backward()

            optimizer_discriminator.step()
            for p in feature_discriminator.parameters():
                p.data.clamp_(-0.1, 0.1)

            for param in feature_discriminator.parameters():
                param.requires_grad = False

            if batch_idx % args['n_critic'] == 0:

                eeg_eog_disc_pred = feature_discriminator(eeg_eog_feature.detach(), (e11.detach(), e22.detach()))  # batch, 20 ,1
                eog_noise_disc_pred1 = feature_discriminator(noise_eog_feature1, (e11, e22))  # batch, 20 ,1
                eog_noise_disc_pred2 = feature_discriminator(noise_eog_feature2, (e11_freq, e22_freq))  # batch, 20 ,1

                # prepare fake labels
                fake_eeg_eog_label = torch.zeros(size=(eeg_eog_feature.size(0),
                                                       eeg_eog_feature.size(1))).long().to(device)
                fake_eog_noise_label = torch.ones(size=(noise_eog_feature1.size(0),
                                                        noise_eog_feature1.size(1))).long().to(device)
                # Compute Adversarial Loss
                loss_adv1 = discriminator_criterion(torch.cat((eeg_eog_disc_pred, eog_noise_disc_pred1), 0).squeeze(),
                                                   torch.cat((fake_eeg_eog_label.float(), fake_eog_noise_label.float()), 0))

                loss_adv2 = discriminator_criterion(torch.cat((eeg_eog_disc_pred, eog_noise_disc_pred2), 0).squeeze(),
                                                   torch.cat((fake_eeg_eog_label.float(), fake_eog_noise_label.float()),
                                                             0))

                loss_adv = 0.5*loss_adv1 + 0.5*loss_adv2
                loss_classifier1 = classifier_criterion(noise_pred1, label.long())
                loss_classifier2 = classifier_criterion(noise_pred2, label.long())

                loss_classifier = 0.5*loss_classifier1 + 0.5*loss_classifier2

                total_loss = args["classifier_2"]*loss_classifier + args["adversarial_w"]*loss_adv

                optimizer_generator.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(feature_generator1.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(feature_generator2.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(sleep_classifier1.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(sleep_classifier2.parameters(), 1.0)

                optimizer_generator.step()

                running_loss += total_loss.item()

            disc_running_loss += loss_disc.item()
            if batch_idx % 10 == 9:  # 输出每次的平均loss
                print('\n [%d,  %5d] D_loss: %.3f  G_loss: %.3f' %
                      (epoch, batch_idx+1, disc_running_loss / 10, running_loss / 10))
                running_loss = 0.0
                disc_running_loss = 0.0


        if epoch % 1 == 0:
            print(f" KFold:{args['Fold']}-------------epoch{epoch}---Multi_Modality_Val------------------------------")
            report = step2_dev((feature_generator1, feature_generator2,
                                sleep_classifier1, sleep_classifier2),
                                val_dl, args, model_param)
            total_acc.append(report[0])
            total_f1.append(report[1])

        if total_acc[-1] > best_acc:
            best_acc = total_acc[-1]
            best_f1 = total_f1[-1]
            best_epoch = epoch
        print("dev_acc:", total_acc)
        print("tmp best acc:", best_acc, best_f1, best_epoch)
        print("dev_macro_f1:", total_f1)
    else:
        print(f"Step2: Best Epoch:{best_epoch}  Best ACC:{best_acc}  Best F1:{best_f1}")
        args['cross_acc'].append(best_acc)
        args['cross_f1'].append(best_f1)


def step2_dev(model, val_dl, args, model_param):
    """
    :param model: (feature_extractor_eog_noise, feature_generator, sleep_classifier)
    :param val_dl: Val Set Dataloader
    :param args: Val parameters
    :param model_param: Model Parameters
    :return: report: tuple(acc, macro_f1)
    """
    if type(model) == tuple:
        model[0].eval()
        model[1].eval()
        model[2].eval()
        model[3].eval()

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
            e11 = eog[:, 0, :].view(batch_size * length, 1, epoch_size)
            e22 = eog[:, 1, :].view(batch_size * length, 1, epoch_size)

            e11_freq = torch.abs(torch.fft.fft(e11))
            e22_freq = torch.abs(torch.fft.fft(e22))

            noise_eog_feature1 = model[0]((e11, e22), gaussian_noise)
            noise_eog_feature2 = model[1]((e11_freq, e22_freq), gaussian_noise)

            # noise_eog_feature = noise_eog_feature1 + noise_eog_feature2
            noise_eog_feature = args["alpha"]*noise_eog_feature1 + (1-args["alpha"])*noise_eog_feature2

            prediction1 = model[2](noise_eog_feature)
            prediction2 = model[3](noise_eog_feature)

            prediction = 0.5*prediction1 + 0.5*prediction2

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

        report = (acc, macro_f1)
        return report
