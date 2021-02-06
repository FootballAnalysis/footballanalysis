from collections import ChainMap
import argparse
import os
import random
import sys
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import copy
import matplotlib.pyplot as plt
from .spec_aug import freq_mask, time_mask, time_warp
from . import model as mod
from .manage_audio import AudioPreprocessor

AUG_MODE = True
CRITERION = 'CE'
EPS = 0.2

class ConfigBuilder(object):
    def __init__(self, *default_configs):
        self.default_config = ChainMap(*default_configs)

    def build_argparse(self):
        parser = argparse.ArgumentParser()
        for key, value in self.default_config.items():
            key = "--{}".format(key)
            if isinstance(value, tuple):
                parser.add_argument(key, default=list(value), nargs=len(value), type=type(value[0]))
            elif isinstance(value, list):
                parser.add_argument(key, default=value, nargs="+", type=type(value[0]))
            elif isinstance(value, bool) and not value:
                parser.add_argument(key, action="store_true")
            else:
                parser.add_argument(key, default=value, type=type(value))
        return parser

    def config_from_argparse(self, parser=None):
        if not parser:
            parser = self.build_argparse()
        args = vars(parser.parse_known_args()[0])
        return ChainMap(args, self.default_config)

def print_eval(name, scores, labels, loss, cm = None, end="\n"):
    batch_size = labels.size(0)
    if cm is not None:
        t_max = torch.max(scores, 1)[1].view(batch_size).data
        for (p, l) in zip(t_max, labels):
            cm[l][p] += 1
    accuracy = (torch.max(scores, 1)[1].view(batch_size).data == labels.data).float().sum() / batch_size
    loss = loss.item()
    print("{} accuracy: {:>5}, loss: {:<25}".format(name, accuracy, loss), end=end)
    return accuracy.item()

def set_seed(config):
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if not config["no_cuda"]:
        torch.cuda.manual_seed(seed)
    random.seed(seed)

def reduce_loss(loss):
    return loss.mean()

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps = 0.1):
        super().__init__()
        self.eps = eps

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = nn.functional.log_softmax(output, dim = -1)
        loss = reduce_loss(-log_preds.sum(dim=-1))
        nll = nn.functional.nll_loss(log_preds, target, reduction = 'mean')
        return (1-self.eps)*nll+self.eps*(loss/c)

def evaluate(config, model=None, test_loader=None):
    if not test_loader:
        _, _, test_set = mod.SpeechDataset.splits(config)
        test_loader = data.DataLoader(
            test_set,
            batch_size=len(test_set),
            collate_fn=test_set.collate_fn)
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
    if not model:
        model = config["model_class"](config)
        model.load(config["input_file"])
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()
    model.eval()
    if CRITERION == 'CE':
        criterion = nn.CrossEntropyLoss()
    elif CRITERION == 'LSCE':
        criterion = LabelSmoothingCrossEntropy(EPS)
    results = []
    total = 0
    cm = np.zeros((config['n_labels'],config['n_labels']))
    for model_in, labels in test_loader:
        model_in = Variable(model_in, requires_grad=False)
        if not config["no_cuda"]:
            model_in = model_in.cuda()
            labels = labels.cuda()
        scores = model(model_in)
        labels = Variable(labels, requires_grad=False)
        loss = criterion(scores, labels)
        results.append(print_eval("test", scores, labels, loss,cm) * model_in.size(0))
        total += model_in.size(0)
    print(cm)
    print("final test accuracy: {}".format(sum(results) / total))

def train(config, print_config=True):
    if print_config:
        print('|------------CONFIG PARAMS------------|')
        for (k, v) in zip(config.keys(), config.values()):
            print(f'param: {k}      --->        value:{v}')
    output_dir = os.path.dirname(os.path.abspath(config["output_file"]))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_set, dev_set, _ = mod.SpeechDataset.splits(config)
    model = config["model_class"](config)
    if config["input_file"]:
        model.load(config["input_file"])
    print('using cuda:', not config["no_cuda"])
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()

        print(next(model.parameters()).device)
    print('|------------MODEL STRUCTURE------------|')
    print(f'number of Param. : {sum(p.numel() for p in model.parameters())}, Trainable ={sum(p.numel() for p in model.parameters() if p.requires_grad == True)}')
    print(model)
    # optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"][0], weight_decay=config["weight_decay"])
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"][0], nesterov=config["use_nesterov"], weight_decay=config["weight_decay"], momentum=config["momentum"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["n_epochs"], eta_min = 0.001)

    schedule_steps = config["schedule"]
    schedule_steps.append(np.inf)
    sched_idx = 0
    # criterion = nn.CrossEntropyLoss()
    if CRITERION == 'CE':
        criterion = nn.CrossEntropyLoss()
    elif CRITERION == 'LSCE':
        criterion = LabelSmoothingCrossEntropy(EPS)
    max_acc = 0
    sampler = data.WeightedRandomSampler(train_set.data_probs,int(train_set.len), replacement = True )
    train_loader = data.DataLoader(
        train_set,
        batch_size=config["batch_size"],
        drop_last=True,
        collate_fn=train_set.collate_fn,
        sampler = sampler)
    dev_loader = data.DataLoader(
        dev_set,
        batch_size=min(len(dev_set), 64),
        shuffle=False,
        collate_fn=dev_set.collate_fn)
    train_accs, dev_accs = [0], [0]
    dev_sv = -1
    # batch_number = (train_set.len//config["batch_size"])+1
    batch_number = len(train_loader)
    for epoch_idx in range(config["n_epochs"]):
        step_no = 0
        cm = np.zeros((config['n_labels'],config['n_labels']))
        accs = []
        print(f'***** Learning rate = {scheduler.get_last_lr()}')
        for batch_idx, (model_in, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            if not config["no_cuda"]:
                model_in = model_in.cuda()
                labels = labels.cuda()
            # print(model_in.shape)
            #AUGMENT
            if AUG_MODE == True:
                for i in range(len(model_in)):
                    # print(model_in[i].shape,'first')
                    # print(model_in[i].permute(1,0).shape,'permute')
                    temp = torch.unsqueeze(model_in[i].permute(1,0),0)
                    if abs(np.random.random())<= 0.5:
                        temp = time_warp(temp, W = 15)
                    if abs(np.random.random())<= 0.5:
                        temp = freq_mask(temp ,F = 6)
                    if abs(np.random.random())<= 0.5:
                        temp = time_mask(temp ,T = 25)
                    model_in[i] = temp.squeeze().permute(1,0)
                    if torch.isnan(model_in).sum()>=1:
                        print('collate_fn')
                    # print(model_in[i].shape,'result')
                    # raise NotImplementedError

            model_in = Variable(model_in, requires_grad=False)
            scores = model(model_in)
            labels = Variable(labels, requires_grad=False)
            # print(model_in)
            # print(scores, labels)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            step_no += 1
            if step_no > schedule_steps[sched_idx]:
                sched_idx += 1
                print("changing learning rate to {}".format(config["lr"][sched_idx]))
                optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"][sched_idx],
                    nesterov=config["use_nesterov"], momentum=config["momentum"], weight_decay=config["weight_decay"])
            accs.append(print_eval(f"epoch:{epoch_idx + 1} /{config['n_epochs']} -->train step {step_no}/{batch_number}, _audio_cache = {train_set._audio_cache.n_keys}", scores,
                       labels, loss, cm))
        avg_acc = np.mean(accs)
        train_accs.append(avg_acc)
        print("final train accuracy: {}".format(avg_acc))
        print(cm)
        np.save(config["output_file"].replace('.pt','_train_cm.npy'),np.array(cm))
        if epoch_idx % config["dev_every"] == config["dev_every"] - 1:
            model.eval()
            accs = []
            cm = np.zeros((config['n_labels'],config['n_labels']))
            for model_in, labels in dev_loader:
                model_in = Variable(model_in, requires_grad=False)
                if not config["no_cuda"]:
                    model_in = model_in.cuda()
                    labels = labels.cuda()
                scores = model(model_in)
                labels = Variable(labels, requires_grad=False)
                loss = criterion(scores, labels)
                accs.append(print_eval(f"dev,   _audio_cache = {dev_set._audio_cache.n_keys}", scores, labels, loss, cm))
            avg_acc = np.mean(accs)
            dev_accs.append(avg_acc)
            print("final dev accuracy: {}".format(avg_acc))
            print(cm)
            np.save(config["output_file"].replace('.pt','_dev_cm.npy'),np.array(cm))
            if avg_acc > max_acc:
                print("saving best model...")
                max_acc = avg_acc
                model.save(config["output_file"])
                dev_sv = epoch_idx
                best_model=copy.deepcopy(model)
        plt.plot(np.arange(0,len(train_accs)),train_accs, label = 'train')
        if len(dev_accs) > 0:
            plt.plot(np.arange(0,len(train_accs),config["dev_every"]),dev_accs, label = 'dev')
        if dev_sv != -1:
            plt.plot([dev_sv+1,dev_sv+1], [0,1], label = 'save model')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('Accuracy')
        plt.title(config["output_file"])
        plt.savefig(config["output_file"].replace('.pt','.png'))
        plt.show(block = False)
        plt.close()
        np.save(config["output_file"].replace('.pt','_dev.npy'),np.array(dev_accs))
        np.save(config["output_file"].replace('.pt','_train.npy'),np.array(train_accs))
        scheduler.step()
    # evaluate(config, best_model, test_loader)

def main():
    output_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "model", "model.pt")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[x.value for x in list(mod.ConfigType)], default="cnn-trad-pool2", type=str)
    config, _ = parser.parse_known_args()

    global_config = dict(no_cuda=False, n_epochs=500, lr=[0.001], schedule=[np.inf], batch_size=64, dev_every=10, seed=0,
        use_nesterov=False, input_file="", output_file=output_file, gpu_no=1, cache_size=32768, momentum=0.9, weight_decay=0.00001)
    mod_cls = mod.find_model(config.model)
    builder = ConfigBuilder(
        mod.find_config(config.model),
        mod.SpeechDataset.default_config(),
        global_config)
    parser = builder.build_argparse()
    parser.add_argument("--type", choices=["train", "eval"], default="train", type=str)
    config = builder.config_from_argparse(parser)
    config["model_class"] = mod_cls
    config["gpu_no"] = 0
    config["no_cuda"] = False
    # config["input_file"] = 'model/model_3sec.pt'
    set_seed(config)
    if config["type"] == "train":
        train(config)
    elif config["type"] == "eval":
        evaluate(config)

if __name__ == "__main__":
    main()

