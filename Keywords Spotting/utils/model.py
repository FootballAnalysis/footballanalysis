from enum import Enum
import hashlib
import math
import os
import random
import re
from chainmap import ChainMap
from torch.autograd import Variable
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from efficientnet_pytorch import EfficientNet
from .manage_audio import AudioPreprocessor


class SimpleCache(dict):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit
        self.n_keys = 0

    def __setitem__(self, key, value):
        if key in self.keys():
            super().__setitem__(key, value)
        elif self.n_keys < self.limit:
            self.n_keys += 1
            super().__setitem__(key, value)
        return value

class ConfigType(Enum):
    CNN_TRAD_POOL2 = "cnn-trad-pool2" # default full model (TF variant)
    CNN_ONE_STRIDE1 = "cnn-one-stride1" # default compact model (TF variant)
    CNN_ONE_FPOOL3 = "cnn-one-fpool3"
    CNN_ONE_FSTRIDE4 = "cnn-one-fstride4"
    CNN_ONE_FSTRIDE8 = "cnn-one-fstride8"
    CNN_TPOOL2 = "cnn-tpool2"
    CNN_TPOOL3 = "cnn-tpool3"
    CNN_TSTRIDE2 = "cnn-tstride2"
    CNN_TSTRIDE4 = "cnn-tstride4"
    CNN_TSTRIDE8 = "cnn-tstride8"
    RES15 = "res15"
    RES26 = "res26"
    RES8 = "res8"
    RES15_NARROW = "res15-narrow"
    RES8_NARROW = "res8-narrow"
    RES26_NARROW = "res26-narrow"
    EFFB0 = "eff0"
    EFFB1 = "eff1"
    EFFB2 = "eff2"

def find_model(conf):
    if isinstance(conf, ConfigType):
        conf = conf.value
    if conf.startswith("res"):
        return SpeechResModel
    elif conf.startswith("eff"):
        return SpeechEffModel
    else:
        return SpeechModel

def find_config(conf):
    if isinstance(conf, ConfigType):
        conf = conf.value
    return _configs[conf]

def truncated_normal(tensor, std_dev=0.01):
    tensor.zero_()
    tensor.normal_(std=std_dev)
    while torch.sum(torch.abs(tensor) > 2 * std_dev) > 0:
        t = tensor[torch.abs(tensor) > 2 * std_dev]
        t.zero_()
        tensor[torch.abs(tensor) > 2 * std_dev] = torch.normal(t, std=std_dev)

class SerializableModule(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))


class SpeechEffModel(SerializableModule):
    def __init__(self, config):
        super().__init__()
        n_labels = config["n_labels"]
        eff_n = config['n_layers']
        self.model = EfficientNet.from_pretrained(f'efficientnet-b{eff_n}',num_classes=n_labels, in_channels = 1 )
        if eff_n == 0:
            block = '_blocks.15'
        else:
            block = '_None'
        for name,param in self.model.named_parameters():
            param.requires_grad = False
            # train bn layers as well as last fully connected and conv head
            if 'bn' in name or '_fc' in name or block in name or '_conv_head.weight' in name : # only train bn layer and last two layers
                param.requires_grad = True
        for name,param in self.model.named_parameters():
           print(str(name), param.requires_grad)
    def forward(self, x):
        x = x.unsqueeze(1)
        return self.model(x)

class SpeechResModel(SerializableModule):
    def __init__(self, config):
        super().__init__()
        n_labels = config["n_labels"]
        n_maps = config["n_feature_maps"]
        self.conv0 = nn.Conv2d(1, n_maps, (3, 3), padding=(1, 1), bias=False)
        if "res_pool" in config:
            self.pool = nn.AvgPool2d(config["res_pool"])

        self.n_layers = n_layers = config["n_layers"]
        dilation = config["use_dilation"]
        if dilation:
            self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=int(2**(i // 3)), dilation=int(2**(i // 3)),
                bias=False) for i in range(n_layers)]
        else:
            self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=1, dilation=1,
                bias=False) for _ in range(n_layers)]
        for i, conv in enumerate(self.convs):
            self.add_module("bn{}".format(i + 1), nn.BatchNorm2d(n_maps, affine=False))
            self.add_module("conv{}".format(i + 1), conv)
        self.output = nn.Linear(n_maps, n_labels)

    def forward(self, x):
        x = x.unsqueeze(1)
        for i in range(self.n_layers + 1):
            y = F.relu(getattr(self, "conv{}".format(i))(x))
            if i == 0:
                if hasattr(self, "pool"):
                    y = self.pool(y)
                old_x = y
            if i > 0 and i % 2 == 0:
                x = y + old_x
                old_x = x
            else:
                x = y
            if i > 0:
                x = getattr(self, "bn{}".format(i))(x)
        x = x.view(x.size(0), x.size(1), -1) # shape: (batch, feats, o3)
        x = torch.mean(x, 2)
        return self.output(x)

class SpeechModel(SerializableModule):
    def __init__(self, config):
        super().__init__()
        n_labels = config["n_labels"]
        n_featmaps1 = config["n_feature_maps1"]
        conv1_size = config["conv1_size"] # (time, frequency)
        conv1_pool = config["conv1_pool"]
        conv1_stride = tuple(config["conv1_stride"])
        dropout_prob = config["dropout_prob"]
        width = config["width"]
        height = config["height"]
        self.conv1 = nn.Conv2d(1, n_featmaps1, conv1_size, stride=conv1_stride)
        tf_variant = config.get("tf_variant")
        self.tf_variant = tf_variant
        if tf_variant:
            truncated_normal(self.conv1.weight.data)
            self.conv1.bias.data.zero_()
        self.pool1 = nn.MaxPool2d(conv1_pool)

        x = Variable(torch.zeros(1, 1, height, width), volatile=True)
        x = self.pool1(self.conv1(x))
        conv_net_size = x.view(1, -1).size(1)
        last_size = conv_net_size

        if "conv2_size" in config:
            conv2_size = config["conv2_size"]
            conv2_pool = config["conv2_pool"]
            conv2_stride = tuple(config["conv2_stride"])
            n_featmaps2 = config["n_feature_maps2"]
            self.conv2 = nn.Conv2d(n_featmaps1, n_featmaps2, conv2_size, stride=conv2_stride)
            if tf_variant:
                truncated_normal(self.conv2.weight.data)
                self.conv2.bias.data.zero_()
            self.pool2 = nn.MaxPool2d(conv2_pool)
            x = self.pool2(self.conv2(x))
            conv_net_size = x.view(1, -1).size(1)
            last_size = conv_net_size
        if not tf_variant:
            self.lin = nn.Linear(conv_net_size, 32)

        if "dnn1_size" in config:
            dnn1_size = config["dnn1_size"]
            last_size = dnn1_size
            if tf_variant:
                self.dnn1 = nn.Linear(conv_net_size, dnn1_size)
                truncated_normal(self.dnn1.weight.data)
                self.dnn1.bias.data.zero_()
            else:
                self.dnn1 = nn.Linear(32, dnn1_size)
            if "dnn2_size" in config:
                dnn2_size = config["dnn2_size"]
                last_size = dnn2_size
                self.dnn2 = nn.Linear(dnn1_size, dnn2_size)
                if tf_variant:
                    truncated_normal(self.dnn2.weight.data)
                    self.dnn2.bias.data.zero_()
        self.output = nn.Linear(last_size, n_labels)
        if tf_variant:
            truncated_normal(self.output.weight.data)
            self.output.bias.data.zero_()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = F.relu(self.conv1(x.unsqueeze(1))) # shape: (batch, channels, i1, o1)
        x = self.dropout(x)
        x = self.pool1(x)
        if hasattr(self, "conv2"):
            x = F.relu(self.conv2(x)) # shape: (batch, o1, i2, o2)
            x = self.dropout(x)
            x = self.pool2(x)
        x = x.view(x.size(0), -1) # shape: (batch, o3)
        if hasattr(self, "lin"):
            x = self.lin(x)
        if hasattr(self, "dnn1"):
            x = self.dnn1(x)
            if not self.tf_variant:
                x = F.relu(x)
            x = self.dropout(x)
        if hasattr(self, "dnn2"):
            x = self.dnn2(x)
            x = self.dropout(x)
        return self.output(x)

class DatasetType(Enum):
    TRAIN = 0
    DEV = 1
    TEST = 2

class SpeechDataset(data.Dataset):
    LABEL_SILENCE = "__silence__"
    LABEL_UNKNOWN = "__unknown__"
    bg_noise_audio = []
    bg_audio = []
    def __init__(self, data, set_type, config):
        super().__init__()
        self.i=0
        self.without_word = 0.25
        self.SR = 16000
        self.PADD = 0.15
        self.LENGTH = 2
        self.BOUND = 0.2
        self.VOICE_WINDOW = np.clip(librosa.filters.get_window(('kaiser', 1.5), self.SR*1.25)+np.abs(np.random.normal(0,0.1)),0,1.1)
        self.BACKGROUND_WINDOW = 1-librosa.filters.get_window(('kaiser', 2.5), self.SR*(1+2*self.PADD))+0.05
        self.audio_files = list(data.keys())
        self.set_type = set_type
        self.audio_labels = list(data.values())
        config["bg_noise_files"] = list(filter(lambda x: x.endswith("npy"), config.get("bg_noise_files", [])))
        config["bg_files"] = list(filter(lambda x: x.endswith("npy"), config.get("bg_files", [])))

        self.unknown_prob = config["unknown_prob"]
        self.silence_prob = config["silence_prob"]
        self.noise_prob = config["noise_prob"]
        self.input_length = config["input_length"]
        self.timeshift_ms = config["timeshift_ms"]
        self._audio_cache = SimpleCache(config["cache_size"])
        self._file_cache = SimpleCache(config["cache_size"])
        n_unk = len(list(filter(lambda x: x == 1, self.audio_labels)))
        self.n_silence = int(self.silence_prob * (len(self.audio_labels) - n_unk))
        self.audio_processor = AudioPreprocessor(n_mels=config["n_mels"], n_dct_filters=config["n_dct_filters"], hop_ms=10)
        self.audio_preprocess_type = config["audio_preprocess_type"]
        total_lbl = list(data.values()) +[0]*self.n_silence
        uniq_labels = sorted(list(set(total_lbl)))
        uniq_idx = {k:v for v,k in enumerate(uniq_labels)}
        labels_count = [(np.array(total_lbl)==l).sum() for l in uniq_labels]
        if len(labels_count)>2:
            self.len = max(labels_count[2:])*len(labels_count)
        else:
            self.len = 0
        # for weighted random sampler
        self.data_probs = [1/labels_count[uniq_idx[l]] for l in total_lbl]

        if len(SpeechDataset.bg_noise_audio) == 0:
            SpeechDataset.bg_noise_audio = [np.load(file) for file in config["bg_noise_files"]]
            if len(SpeechDataset.bg_noise_audio) == 0:
                SpeechDataset.bg_noise_audio = [np.zeros(self.input_length)]
            print(f'# of background noise: {len(SpeechDataset.bg_noise_audio)}')
        if len(SpeechDataset.bg_audio) == 0:
            SpeechDataset.bg_audio = [np.load(file) for file in config["bg_files"]]
            if len(SpeechDataset.bg_audio) == 0:
                SpeechDataset.bg_audio = [np.zeros(self.input_length*2)]
            print(f'# of background: {len(SpeechDataset.bg_audio)}')

    @staticmethod
    def default_config():
        config = {}
        config["group_speakers_by_id"] = True
        config["silence_prob"] = 0.1
        config["noise_prob"] = 0.8
        config["n_dct_filters"] = 40
        config["input_length"] = 16000
        config["n_mels"] = 40
        config["timeshift_ms"] = 100
        config["unknown_prob"] = 0.1
        config["train_pct"] = 80
        config["dev_pct"] = 10
        config["test_pct"] = 10
        config["wanted_words"] = ["command", "random"]
        config["data_folder"] = "/data/speech_dataset"
        config["audio_preprocess_type"] = "MFCCs"
        return config

    def collate_fn(self, data):
        x = None
        y = []
        for audio_data, label in data:
            if self.audio_preprocess_type == "MFCCs":
                audio_tensor = torch.from_numpy(self.audio_processor.compute_mfccs(audio_data).reshape(1, -1, 40))
                x = audio_tensor if x is None else torch.cat((x, audio_tensor), 0)
            elif self.audio_preprocess_type == "PCEN":
                audio_tensor = torch.from_numpy(np.expand_dims(audio_data, axis=0))
                audio_tensor = self.audio_processor.compute_pcen(audio_tensor)
                x = audio_tensor if x is None else torch.cat((x, audio_tensor), 0)
            y.append(label)
        return x, torch.tensor(y)

    def _timeshift_audio(self, data):
        shift = (16000 * self.timeshift_ms) // 1000
        shift = random.randint(-shift, shift)
        a = -min(0, shift)
        b = max(0, shift)
        data = np.pad(data, (a, b), "constant")
        return data[:len(data) - a] if a else data[b:]

    def _get_bg_voice(self):
        idx = np.random.randint(len(SpeechDataset.bg_audio))
        background_voices = SpeechDataset.bg_audio[idx]
        start_index = np.random.randint(0,len(background_voices)-self.SR*self.LENGTH)
        background = np.copy(background_voices[start_index:start_index+int(self.SR*self.LENGTH)])
        return background

    def _get_bg_noise(self):
        bg_noise = random.choice(SpeechDataset.bg_noise_audio)
        start_idx = random.randint(0, len(bg_noise) - self.input_length - 1)
        bg_noise = np.copy(bg_noise[start_idx:start_idx + self.input_length])
        return bg_noise

    def load_audio(self, example, silence=False, lbl = 0):
        in_len = self.input_length
        bg_noise_multi = random.random() *0.25
        bg_noise = self._get_bg_noise()
        if silence:
            example = "__silence__"
        if lbl == 1 and self.set_type == DatasetType.TRAIN and random.random()< self.without_word:
            return np.clip(bg_noise_multi * bg_noise + self._get_bg_voice(), -1, 1)
        try:
            data =  np.copy(self._audio_cache[example])
            if example != "__silence__" and self.set_type == DatasetType.TRAIN:
                data *= np.clip(self.VOICE_WINDOW[:len(data)]+np.abs(np.random.normal(0,0.02)),0,1.1)
                background = self._get_bg_voice()
                start_index = np.random.randint(int(self.BOUND*self.SR),len(background)-len(self.BACKGROUND_WINDOW)-int(self.BOUND*self.SR))
                background[start_index:start_index+len(self.BACKGROUND_WINDOW)]*= self.BACKGROUND_WINDOW
                start_index += int(self.PADD*self.SR)
                background[start_index:start_index+len(data)] += data
                data = background
            if self.set_type == DatasetType.TRAIN:
                data = np.clip(bg_noise_multi * bg_noise + data, -1, 1)
            return data
        except KeyError:
            bg_noise = random.choice(SpeechDataset.bg_noise_audio)
            start_idx = random.randint(0, len(bg_noise) - in_len - 1)
            bg_noise = np.copy(bg_noise[start_idx:start_idx + in_len])
            if silence:
                data = np.zeros(in_len, dtype=np.float32)
            else:
                file_data = self._file_cache.get(example)
                data = np.load(example) if file_data is None else file_data
                if len(data)>len(self.VOICE_WINDOW):
                    data= data[:len(self.VOICE_WINDOW)]
                self._file_cache[example] = data
            data = np.pad(data, (0, max(0, len(self.VOICE_WINDOW) - len(data))), "constant")
            self._audio_cache[example] = data
            if example != "__silence__":
                data *= np.clip(self.VOICE_WINDOW[:len(data)]+np.abs(np.random.normal(0,0.02)),0,1.1)
                background = self._get_bg_voice()
                start_index = np.random.randint(0,len(background)-len(self.BACKGROUND_WINDOW))
                background[start_index:start_index+len(self.BACKGROUND_WINDOW)]*= self.BACKGROUND_WINDOW
                start_index += int(self.PADD*self.SR)
                background[start_index:start_index+len(data)] += data
                data = background
            if lbl == 1 and not self.set_type == DatasetType.TRAIN and random.random()< self.without_word:
                data = np.clip(bg_noise_multi * bg_noise + self._get_bg_voice(), -1, 1)
            else:
                data = np.clip(bg_noise_multi * bg_noise + data, -1, 1)
            if not self.set_type == DatasetType.TRAIN:
                self._audio_cache[example] = data
            return data

    @classmethod
    def splits(cls, config):
        folder = config["data_folder"]
        wanted_words = config["wanted_words"]
        unknown_prob = config["unknown_prob"]
        train_pct = config["train_pct"]
        dev_pct = config["dev_pct"]
        test_pct = config["test_pct"]
        words = {word: i + 2 for i, word in enumerate(wanted_words)}
        words.update({cls.LABEL_SILENCE:0, cls.LABEL_UNKNOWN:1})
        sets = [{}, {}, {}]
        unknowns = [0] * 3
        bg_noise_files = []
        bg_files = []
        unknown_files = []
        class_count = {}
        for folder_name in os.listdir(folder):
            path_name = os.path.join(folder, folder_name)
            if os.path.isdir(path_name):
                number_item = len(os.listdir(path_name))
                # print(path_name,number_item)
                if folder_name in words:
                    class_count[str(path_name)] = number_item
        # print('class count', class_count)
        dev_count =  int((dev_pct/100)*np.min([v for v in class_count.values()]))
        # print('dev count' , dev_count)
        for folder_name in os.listdir(folder):
            path_name = os.path.join(folder, folder_name)
            is_bg_noise = False
            is_bg = False
            if folder_name != "_background_noise_" and folder_name != "_background_" and folder_name in words:
                dev_data = list(np.random.choice(np.arange(len(os.listdir(path_name))), dev_count, replace = False))
            if os.path.isfile(path_name):
                continue
            if folder_name in words:
                label = words[folder_name]
            elif folder_name == "_background_noise_":
                is_bg_noise = True
            elif folder_name == "_background_":
                is_bg = True
            else:
                label = words[cls.LABEL_UNKNOWN]
            for i,filename in enumerate(os.listdir(path_name)):
                wav_name = os.path.join(path_name, filename)
                if is_bg_noise and os.path.isfile(wav_name):
                    bg_noise_files.append(wav_name)
                    continue
                elif is_bg and os.path.isfile(wav_name):
                    bg_files.append(wav_name)
                    continue
                elif label == words[cls.LABEL_UNKNOWN]:
                    unknown_files.append(wav_name)
                    continue
                if config["group_speakers_by_id"]:
                    hashname = re.sub(r"_nohash_.*$", "", filename)
                max_no_wavs = 2**27 - 1
                bucket = int(hashlib.sha1(hashname.encode()).hexdigest(), 16)
                bucket = (bucket % (max_no_wavs + 1)) * (100. / max_no_wavs)
                test_con = True
                if i in dev_data:
                    tag = DatasetType.DEV
                else:
                    tag = DatasetType.TRAIN
                if test_con and i<5:
                    sets[DatasetType.TEST.value][wav_name] = label
                sets[tag.value][wav_name] = label
        unknowns[0] = len(unknown_files)- dev_count
        unknowns[1] = dev_count
        random.shuffle(unknown_files)
        a = 0
        for i, dataset in enumerate(sets):
            b = a + unknowns[i]
            unk_dict = {u: words[cls.LABEL_UNKNOWN] for u in unknown_files[a:b]}
            dataset.update(unk_dict)
            a = b
        # print(bg_noise_files)
        # print(bg_files)
        train_cfg = ChainMap(dict(bg_noise_files=bg_noise_files,bg_files=bg_files), config)
        test_cfg = ChainMap(dict(bg_noise_files=bg_noise_files,bg_files=bg_files, noise_prob=0), config)
        datasets = (cls(sets[0], DatasetType.TRAIN, train_cfg), cls(sets[1], DatasetType.DEV, test_cfg),
                cls(sets[2], DatasetType.TEST, test_cfg))
        return datasets

    def __getitem__(self, index):
        if index >= len(self.audio_labels):
            return self.load_audio(None,True,0), 0
        return self.load_audio(self.audio_files[index],False,self.audio_labels[index]), self.audio_labels[index]

    def __len__(self):
        return len(self.audio_labels) + self.n_silence

_configs = {
    ConfigType.CNN_TRAD_POOL2.value: dict(dropout_prob=0.5, height=101, width=40, n_labels=4, n_feature_maps1=64,
        n_feature_maps2=64, conv1_size=(20, 8), conv2_size=(10, 4), conv1_pool=(2, 2), conv1_stride=(1, 1),
        conv2_stride=(1, 1), conv2_pool=(1, 1), tf_variant=True),
    ConfigType.CNN_ONE_STRIDE1.value: dict(dropout_prob=0.5, height=101, width=40, n_labels=4, n_feature_maps1=186,
        conv1_size=(101, 8), conv1_pool=(1, 1), conv1_stride=(1, 1), dnn1_size=128, dnn2_size=128, tf_variant=True),
    ConfigType.CNN_TSTRIDE2.value: dict(dropout_prob=0.5, height=101, width=40, n_labels=4, n_feature_maps1=78,
        n_feature_maps2=78, conv1_size=(16, 8), conv2_size=(9, 4), conv1_pool=(1, 3), conv1_stride=(2, 1),
        conv2_stride=(1, 1), conv2_pool=(1, 1), dnn1_size=128, dnn2_size=128),
    ConfigType.CNN_TSTRIDE4.value: dict(dropout_prob=0.5, height=101, width=40, n_labels=4, n_feature_maps1=100,
        n_feature_maps2=78, conv1_size=(16, 8), conv2_size=(5, 4), conv1_pool=(1, 3), conv1_stride=(4, 1),
        conv2_stride=(1, 1), conv2_pool=(1, 1), dnn1_size=128, dnn2_size=128),
    ConfigType.CNN_TSTRIDE8.value: dict(dropout_prob=0.5, height=101, width=40, n_labels=4, n_feature_maps1=126,
        n_feature_maps2=78, conv1_size=(16, 8), conv2_size=(5, 4), conv1_pool=(1, 3), conv1_stride=(8, 1),
        conv2_stride=(1, 1), conv2_pool=(1, 1), dnn1_size=128, dnn2_size=128),
    ConfigType.CNN_TPOOL2.value: dict(dropout_prob=0.5, height=101, width=40, n_labels=4, n_feature_maps1=94,
        n_feature_maps2=94, conv1_size=(21, 8), conv2_size=(6, 4), conv1_pool=(2, 3), conv1_stride=(1, 1),
        conv2_stride=(1, 1), conv2_pool=(1, 1), dnn1_size=128, dnn2_size=128),
    ConfigType.CNN_TPOOL3.value: dict(dropout_prob=0.5, height=101, width=40, n_labels=4, n_feature_maps1=94,
        n_feature_maps2=94, conv1_size=(15, 8), conv2_size=(6, 4), conv1_pool=(3, 3), conv1_stride=(1, 1),
        conv2_stride=(1, 1), conv2_pool=(1, 1), dnn1_size=128, dnn2_size=128),
    ConfigType.CNN_ONE_FPOOL3.value: dict(dropout_prob=0.5, height=101, width=40, n_labels=4, n_feature_maps1=54,
        conv1_size=(101, 8), conv1_pool=(1, 3), conv1_stride=(1, 1), dnn1_size=128, dnn2_size=128),
    ConfigType.CNN_ONE_FSTRIDE4.value: dict(dropout_prob=0.5, height=101, width=40, n_labels=4, n_feature_maps1=186,
        conv1_size=(101, 8), conv1_pool=(1, 1), conv1_stride=(1, 4), dnn1_size=128, dnn2_size=128),
    ConfigType.CNN_ONE_FSTRIDE8.value: dict(dropout_prob=0.5, height=101, width=40, n_labels=4, n_feature_maps1=336,
        conv1_size=(101, 8), conv1_pool=(1, 1), conv1_stride=(1, 8), dnn1_size=128, dnn2_size=128),
    ConfigType.RES15.value: dict(n_labels=12, use_dilation=True, n_layers=13, n_feature_maps=45),
    ConfigType.RES8.value: dict(n_labels=12, n_layers=6, n_feature_maps=45, res_pool=(4, 3), use_dilation=False),
    ConfigType.RES26.value: dict(n_labels=12, n_layers=24, n_feature_maps=45, res_pool=(2, 2), use_dilation=False),
    ConfigType.RES15_NARROW.value: dict(n_labels=12, use_dilation=True, n_layers=13, n_feature_maps=19),
    ConfigType.RES8_NARROW.value: dict(n_labels=12, n_layers=6, n_feature_maps=19, res_pool=(4, 3), use_dilation=False),
    ConfigType.RES26_NARROW.value: dict(n_labels=12, n_layers=24, n_feature_maps=19, res_pool=(2, 2), use_dilation=False),
    ConfigType.EFFB0.value: dict(n_labels=12, n_layers=0, n_feature_maps=19, res_pool=(2, 2), use_dilation=False),
    ConfigType.EFFB1.value: dict(n_labels=12, n_layers=1, n_feature_maps=19, res_pool=(2, 2), use_dilation=False),
    ConfigType.EFFB2.value: dict(n_labels=12, n_layers=2, n_feature_maps=19, res_pool=(2, 2), use_dilation=False),
}
