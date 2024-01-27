
import os
import torch
from importlib import import_module
import torch.nn as nn
def init_model(model, args):

    device = torch.device(args.device)
    model.to(device)


    return model#, optimizer, loss_scaler

def get_model(args):
    module = import_module('models.' + args.model.lower()) #引入 可以使用该文件的所有函数 变量信息  import module模块十分的常用啊 用来找模型
    #print("初始化的模型：",init_model(module.make_model(args), args))
    return init_model(module.make_model(args), args)
