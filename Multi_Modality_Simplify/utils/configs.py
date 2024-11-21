# Time : 2023/2/16 11:44
# Author : 小霸奔
# FileName: configs.p
class ModelConfig(object):
    def __init__(self):
        self.EegNum = 6
        self.EogNum = 2
        self.EmgNum = 3
        self.ConvDrop = 0.1
        self.EncoderParam = EncoderConfig()
        self.SleepMlpParam = SleepMlpParam()
        self.DiscriminatorParam = DiscriminatorParam()
        self.NumClasses = 5
        self.ClassNames = ['W', 'N1', 'N2', 'N3', 'REM']
        self.SeqLength = 20
        self.BatchSize = 32
        self.EpochLength = 3000


class EncoderConfig(object):
    def __init__(self):
        self.n_head = 8
        self.d_model = 512
        self.layer_num = 3
        self.drop = 0.1


class SleepMlpParam(object):
    def __init__(self):
        self.drop = 0.1
        self.first_linear = [512, 256]
        self.second_linear = [256, 128]
        self.out_linear = [128, 5]


class DiscriminatorParam(object):
    def __init__(self):
        self.first_linear = [512, 256]
        self.second_linear = [256, 128]
        self.out_linear = [128, 1]


