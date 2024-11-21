# Time : 2022/3/8 16:34
# Author : 小霸奔
# FileName: raw_data_CNN.p
import torch.nn as nn
import torch
from model.util_block import SRMLayer, MultiHeadAttentionBlock, EEGModalityFusion, EOGModalityFusion, AllModalityFusion
from model.util_block import EMGModalityFusion
from utils.configs import ModelConfig

ModelParam = ModelConfig()


class FeatureExtractor(nn.Module):
    def __init__(self, ):
        self.drop = ModelParam.ConvDrop
        super(FeatureExtractor, self).__init__()
        self.time_sequential = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(self.drop),

            nn.Conv1d(64, 128, kernel_size=8),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Conv1d(128, 256, kernel_size=8),
            nn.BatchNorm1d(256),
            nn.GELU(),

            nn.Conv1d(256, 512, kernel_size=8),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )

        self.freq_sequential = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=200, stride=25, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(self.drop),

            nn.Conv1d(64, 128, kernel_size=6),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Conv1d(128, 256, kernel_size=6),
            nn.BatchNorm1d(256),
            nn.GELU(),

            nn.Conv1d(256, 512, kernel_size=6),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.avg = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        batch = x.shape[0]//ModelParam.SeqLength
        x1 = self.time_sequential(x)
        x2 = self.freq_sequential(x)
        x = torch.concat((x1, x2), dim=2)
        x = self.avg(x).view(batch*ModelParam.SeqLength, 1, 512)
        return x


class SleepMLP(nn.Module):
    def __init__(self):
        super(SleepMLP, self).__init__()
        self.dropout_rate = ModelParam.SleepMlpParam.drop
        self.sleep_stage_mlp = nn.Sequential(
            nn.Linear(512, ModelParam.SleepMlpParam.first_linear[1]),
            nn.Dropout(self.dropout_rate),
            nn.GELU(),
            nn.Linear(ModelParam.SleepMlpParam.second_linear[0], ModelParam.SleepMlpParam.second_linear[1]),
            nn.Dropout(self.dropout_rate),
            nn.GELU(),
        )
        self.sleep_stage_classifier = nn.Linear(ModelParam.SleepMlpParam.out_linear[0],
                                                ModelParam.SleepMlpParam.out_linear[1], bias=False)

    def forward(self, x):
        x = self.sleep_stage_mlp(x)
        x = self.sleep_stage_classifier(x)
        x = x.permute(0, 2, 1)
        return x

class SleepMLP2(nn.Module):
    def __init__(self):
        super(SleepMLP2, self).__init__()
        self.dropout_rate = ModelParam.SleepMlpParam.drop
        self.sleep_stage_mlp = nn.Sequential(
            nn.Linear(512, ModelParam.SleepMlpParam.first_linear[1]),
            nn.Dropout(self.dropout_rate),
            nn.GELU(),
            nn.Linear(ModelParam.SleepMlpParam.second_linear[0], ModelParam.SleepMlpParam.second_linear[1]),
            nn.Dropout(self.dropout_rate),
            nn.GELU(),
        )
        self.sleep_stage_classifier = nn.Linear(ModelParam.SleepMlpParam.out_linear[0],
                                                ModelParam.SleepMlpParam.out_linear[1], bias=False)

        for m in self.modules():
            if isinstance(m, (torch.nn.Linear)):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.sleep_stage_mlp(x)
        x = self.sleep_stage_classifier(x)
        x = x.permute(0, 2, 1)
        return x

class AttFuEncode(torch.nn.Module):
    def __init__(self):
        super(AttFuEncode, self).__init__()
        self.encoder = MultiHeadAttentionBlock(ModelParam.EncoderParam.d_model,
                                               ModelParam.EncoderParam.layer_num,
                                               ModelParam.EncoderParam.drop,
                                               ModelParam.EncoderParam.n_head)
        self.channel_num = ModelParam.EegNum+ModelParam.EogNum
        self.channel_attention_module = SRMLayer(self.channel_num)
        self.eeg_fusion = EEGModalityFusion(ModelParam.EegNum)
        self.eog_fusion = EOGModalityFusion(ModelParam.EogNum)
        self.all_fusion = AllModalityFusion()
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.tanh = nn.Tanh()



    def forward(self, f4, f3, c4, c3, o1, o2, e1, e2):
        batch = f4.shape[0] // ModelParam.SeqLength
        x = torch.concat((f4, f3, c4, c3, o1, o2, e1, e2), dim=1)
        channel_attention = self.channel_attention_module(x)
        x = x.mul(channel_attention)

        eeg = self.eeg_fusion(x[:, :6, :])
        eog = self.eog_fusion(x[:, 6:, :])

        x = self.all_fusion(torch.concat((eeg, eog), dim=-1))  # dim=1 conv dim=2 linear
        x = x.view(batch, ModelParam.SeqLength, -1)

        x = self.tanh(self.encoder(x))

        return x


class Discriminator_ISRUC(nn.Module):
    """
    IDEA1: Conditional GAN
    INPUT: Multi Modality Feature:(batch, 20, 512)
           Conditional Input EOG: (batch*length, 1, 512)
    """
    def __init__(self):
        super(Discriminator_ISRUC, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(ModelParam.DiscriminatorParam.first_linear[0], ModelParam.DiscriminatorParam.first_linear[1]),
            nn.GELU(),
            nn.Linear(ModelParam.DiscriminatorParam.second_linear[0], ModelParam.DiscriminatorParam.second_linear[1]),
            nn.GELU(),
            nn.Linear(ModelParam.DiscriminatorParam.out_linear[0], ModelParam.DiscriminatorParam.out_linear[1]),
            nn.Sigmoid()
        )
        self.ff = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=49, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),

            nn.Conv1d(64, 128, kernel_size=9, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),

            nn.Conv1d(128, 256, kernel_size=9, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),

            nn.Conv1d(256, 512, kernel_size=9, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),
        )
        self.EogLinear = nn.Linear(1024, 512)
        self.fusion = nn.Linear(1024, 512)
        self.avg = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, eog):
        e1 = eog[0]
        e2 = eog[1]
        batch = e1.shape[0]
        e1 = self.ff(e1)
        e1 = self.avg(e1).view(batch, 1, 512)
        e2 = self.ff(e2)
        e2 = self.avg(e2).view(batch, 1, 512)
        eog = self.EogLinear(torch.concat((e1, e2), dim=2))  # batch*length, 1, 512
        eog = eog.view(-1, ModelParam.SeqLength, 512)  # batch, length, 512
        x = self.fusion(torch.concat((x, eog), dim=2))  # batch, length, 1024 -> batch, length, 512
        out = self.layer(x)
        return out


class NoiseEogFusion_ISRUC(nn.Module):
    """
    IDEA1: Conditional GAN
    INPUT: Noise size(batch, length, 512)
           E1 E2 size(batch*length, 1, 512)
           Step1: Concat E1 E2 to Linear and view : (batch, length, 512)
           Step2: Concat EOG Noise and Linear: (batch, length, 1024) -> (batch, length, 512)
           Step3: Fusion Feature to Feedforward  out:(batch, length, 512)
    """
    def __init__(self):
        super(NoiseEogFusion_ISRUC, self).__init__()
        self.fusion1 = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.LeakyReLU(),

            nn.Linear(2048, 512),
            nn.LeakyReLU(),

            nn.Linear(512, 2048),
            nn.LeakyReLU(),

            nn.Linear(2048, 512),
            nn.Tanh()
        )
        self.time_sequential = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=49, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),

            nn.Conv1d(64, 128, kernel_size=9, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),

            nn.Conv1d(128, 256, kernel_size=9, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),

            nn.Conv1d(256, 512, kernel_size=9, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),
        )
        self.eoglinear = nn.Linear(1024, 512)
        self.NoiseGenerate = nn.Linear(100, 512)
        self.encoder = MultiHeadAttentionBlock(ModelParam.EncoderParam.d_model,
                                               ModelParam.EncoderParam.layer_num,
                                               ModelParam.EncoderParam.drop,
                                               ModelParam.EncoderParam.n_head)
        self.tanh = nn.Tanh()
        self.avg = nn.AdaptiveAvgPool1d(1)

    def forward(self, eog, noise):
        """
        :param eog: (e1, e2) size(batch*length, 1, 3000)
        :param noise: size(batch, 20, 512)
        :return: Multi_Modality Feature Generalize by Noise and EOG
        """
        e1 = eog[0]
        e2 = eog[1]
        batch = e1.shape[0] // ModelParam.SeqLength
        e1 = self.time_sequential(e1)
        e1 = self.avg(e1).view(batch * ModelParam.SeqLength, 1, 512)

        e2 = self.time_sequential(e2)
        e2 = self.avg(e2).view(batch * ModelParam.SeqLength, 1, 512)

        eog = self.eoglinear(torch.concat((e1, e2), dim=2))
        eog = eog.view(batch, ModelParam.SeqLength, -1)  # batch, length, 512
        noise = self.NoiseGenerate(noise)
        modality_feature = torch.concat((eog, noise), dim=2)
        modality_feature = self.fusion1(modality_feature)

        modality_feature = self.tanh(self.encoder(modality_feature))
        return modality_feature

