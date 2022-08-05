import os
import numpy as np
import shutil


# <bruno>

def pad(a, div):
    if len(a) % div == 0:
        return a
    ix = range(len(a))
    choices = np.random.choice(ix, size=div - (len(a) % div))
    a.extend([a[c] for c in choices])
    return a


def preprocess(x, y, batch_size):
    x, y = shuffle_together(x, y)
    # x = batch(x, batch_size)
    # y = batch(y, batch_size)
    return np.stack(x), np.stack(y)


def batch(a, batch_size):
    a = pad(a, batch_size)
    out = [a[0:batch_size]]
    for i in range(batch_size, len(a), batch_size):
        ap = a[i: i + batch_size]
        out.append(ap)
    return out


def shuffle_together(a, b):
    assert len(a) == len(b)
    p = list(np.random.permutation(len(a)))
    a = [a[i] for i in p]
    b = [b[i] for i in p]
    return a, b


# </bruno>


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def count_parameters_in_MB(model):
    return (
            np.sum(
                np.prod(v.size())
                for name, v in model.named_parameters()
                if "auxiliary" not in name
            )
            / 1e6
    )


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print("Experiment dir : {}".format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, "scripts"))
        for script in scripts_to_save:
            dst_file = os.path.join(path, "scripts", os.path.basename(script))
            shutil.copyfile(script, dst_file)


def list_var_name(list_of_tensors):
    """输入一个Tensor列表，返回一个包含每个tensor名字的列表
    """
    return [var.name for var in list_of_tensors]


def get_var(list_of_tensors, prefix_name=None):
    """输入一个Tensor列表(可选变量名前缀)，返回[变量名],[变量]两个列表
    """
    if prefix_name is None:
        return list_var_name(list_of_tensors), list_of_tensors
    else:
        specific_tensor = []
        specific_tensor_name = []
        for var in list_of_tensors:
            if var.name.startswith(prefix_name):
                specific_tensor.append(var)
                specific_tensor_name.append(var.name)
        return specific_tensor_name, specific_tensor
