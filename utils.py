# -*- coding: utf-8 -*-

import numpy as np
import time
from graphviz import Digraph
import torch
from torch.autograd import Variable


def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


def save_pr(out_dir, name, epoch, pre, rec, fp_res=None, opt=None):
    if opt is None:
        out = file('{}/{}_{}_PR.txt'.format(out_dir, name, epoch + 1), 'w')
    else:
        out = file('{}/{}_{}_{}_PR.txt'.format(out_dir, name, opt, epoch + 1), 'w')

    if fp_res is not None:
        fp_out = file('{}/{}_{}_FP.txt'.format(out_dir, name, epoch + 1), 'w')
        for idx, r, p in fp_res:
            fp_out.write('{} {} {}\n'.format(idx, r, p))
        fp_out.close()

    for p, r in zip(pre, rec):
        out.write('{} {}\n'.format(p, r))

    out.close()


def eval_metric(true_y, pred_y, pred_p):
    '''
    calculate the precision and recall for p-r curve
    reglect the NA relation
    '''
    assert len(true_y) == len(pred_y)
    positive_num = len([i for i in true_y if i[0] > 0])
    index = np.argsort(pred_p)[::-1]

    tp = 0
    fp = 0
    fn = 0
    all_pre = [0]
    all_rec = [0]
    fp_res = []

    for idx in range(len(true_y)):
        i = true_y[index[idx]]
        j = pred_y[index[idx]]

        if i[0] == 0:  # NA relation
            if j > 0:
                fp_res.append((index[idx], j, pred_p[index[idx]]))
                fp += 1
        else:
            if j == 0:
                fn += 1
            else:
                for k in i:
                    if k == -1:
                        break
                    if k == j:
                        tp += 1
                        break

        if fp + tp == 0:
            precision = 1.0
        else:
            precision = tp * 1.0 / (tp + fp)
        recall = tp * 1.0 / positive_num
        if precision != all_pre[-1] or recall != all_rec[-1]:
            all_pre.append(precision)
            all_rec.append(recall)

    print("tp={}; fp={}; fn={}; positive_num={}".format(tp, fp, fn, positive_num))
    return all_pre[1:], all_rec[1:], fp_res


def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function

    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot
