import os
import sys

import dill
from tensorflow.python.training.tracking.data_structures import ListWrapper

import genotypes
from graphviz import Digraph


def plot(genotype, filename):
    g = Digraph(
        format='png',
        edge_attr=dict(fontsize='20', fontname="sans"),
        node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5',
                       penwidth='2', fontname="sans"),
        engine='dot'
    )
    g.attr(rankdir='LR', pad='0.5,1')

    steps = len(genotype)
    names = [str(i) for i in range(steps)]
    ops = [genotype[i][0] for i in range(steps)]

    g.node("C(k-1)", fillcolor='darkseagreen2')
    g.node("C(k)", fillcolor='palegoldenrod')

    for i in range(steps):
        g.node(names[i], fillcolor='lightblue')

    g.edge("C(k-1)", names[0], fillcolor="gray")

    for i in range(steps-1):
        g.edge(names[i], names[i + 1], label=ops[i], fillcolor="gray")

    g.edge(names[steps-1], "C(k)", ops[steps - 1], fillcolor="gray")

    g.render(filename, cleanup=True)


if __name__ == '__main__':
    genotype = dill.load(
        open("./outputs/Main_Exp/Ext_C1_Commercial/['Restaurant']/0909_1512/1Layers/0/model/genotype.pk", 'rb'))
    plot(genotype.cell, './test')
    # if len(sys.argv) != 2:
    #   print("usage:\n python {} ARCH_NAME".format(sys.argv[0]))
    #   sys.exit(1)

    # genotype_name = sys.argv[1]
    # try:
    #   genotype = eval('genotypes.{}'.format(genotype_name))
    # except AttributeError:
    #   print("{} is not specified in genotypes.py".format(genotype_name))
    #   sys.exit(1)
    # f = open("outputs/train_model/genotype_record_file.txt", 'r')
    # lines = f.readlines()
    #
    # f.close()
    # line = lines[-1]
    # genotype = line[line.find("reduce=") + 7:(line.find("],", line.find("reduce=")) + 1)]
    # genotype_new = []
    # currnt_index = 0
    # while not genotype.find("(", currnt_index) == -1:
    #     left = genotype.find("(", currnt_index)
    #     right = genotype.find(")", currnt_index)
    #     sub = genotype[left + 1:right]
    #     genotype_new.append((sub.split(',')[0][1:-1], int(sub.split(',')[1][1])))
    #     currnt_index = right + 1
    # print(genotype_new)

    # plot(genotype_new, "normal")
    # genotype = genotypes.Genotype(
    #     cell=[('dense', 0), ('dense', 1), ('dense', 2), ('dense', 0), ('dropout', 1), ('dense', 2), ('skip_connect', 1),
    #           ('batch_norm', 4)], concat=[2, 3, 4, 5])
    #
    # for root, dirs, files in os.walk('./outputs/VCAB'):
    #     for filename in files:
    #         if filename == 'genotype.txt':
    #             with open(os.path.join(root, filename), 'r') as file:
    #                 genotype = genotypes.Genotype(file.read())
    #                 print(genotype)
    # if filename == 'model_state.pk':
    #     genotype = Model.load_model(root).genotype
    #     plot(genotype.cell, os.path.join(os.path.dirname(root), 'genotype'))

    # plot(genotype.cell, "freight_wagon_A_1layers")
