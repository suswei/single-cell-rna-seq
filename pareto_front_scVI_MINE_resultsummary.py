import os
import itertools
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

#find the parento front when the goal is to minimize two objectives
def simple_cull(inputPoints, dominates):
    paretoPoints = set()
    candidateRowNr = 0
    dominatedPoints = set()
    while True:
        candidateRow = inputPoints[candidateRowNr]
        inputPoints.remove(candidateRow)
        rowNr = 0
        nonDominated = True
        while len(inputPoints) != 0 and rowNr < len(inputPoints):
            row = inputPoints[rowNr]
            if dominates(candidateRow, row):
                # If it is worse on all features remove the row from the array
                inputPoints.remove(row)
                dominatedPoints.add(tuple(row))
            elif dominates(row, candidateRow):
                nonDominated = False
                dominatedPoints.add(tuple(candidateRow))
                rowNr += 1
            else:
                rowNr += 1

        if nonDominated:
            # add the non-dominated point to the Pareto frontier
            paretoPoints.add(tuple(candidateRow))

        if len(inputPoints) == 0:
            break
    return paretoPoints, dominatedPoints

def dominates(row, candidateRow):
    return sum([row[x] <= candidateRow[x] for x in range(len(row))]) == len(row)


def draw_plot(obj1, obj2, pareto_front_type, type, save_path):

    inputPoints1 = [[obj1[k], obj2[k]] for k in range(len(obj1))]

    paretoPoints1, dominatedPoints1 = simple_cull(inputPoints1, dominates)

    fig = plt.figure()
    dp = np.array(list(dominatedPoints1))
    pp = np.array(list(paretoPoints1))
    plt.scatter(dp[:, 0], dp[:, 1])
    plt.scatter(pp[:, 0], pp[:, 1], color='red')
    if pareto_front_type == 'minibatch':
        plt.title('obj1: std_negative_ELBO, obj2: std_MINE, {}, {}'.format(pareto_front_type, type), fontsize=18)
    elif pareto_front_type == 'full':
        plt.title('obj1: std_negative_ELBO, obj2: MINE, {}, {}'.format(pareto_front_type, type), fontsize=18)
    else:
        plt.title('obj1: be, obj2: {}, {}'.format(pareto_front_type, type), fontsize=18)
    plt.xlabel('obj1', fontsize=16)
    plt.ylabel('obj2', fontsize=16)
    fig.savefig(save_path + '/pareto_front_{}_{}.png'.format(pareto_front_type, type))
    plt.close(fig)

def pareto_front(dataset: str='muris_tabula', confounder: str='batch'):

    dir_path = './result/pareto_front_scVI_MINE/{}/{}'.format(dataset, confounder)
    hyperparameter_config = {
        'scale': [ele/10 for ele in range(0, 11)],
        'MCs': 20*[1]
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results_config_total = pd.DataFrame()
    for i in range(len(hyperparameter_experiments)):
            config_path =  dir_path + '/taskid{}/config.pkl'.format(i)
            results_path = dir_path + '/taskid{}/results.pkl'.format(i)
            if os.path.isfile(config_path) and os.path.isfile(results_path):
                config = pickle.load(open(config_path, "rb"))
                results = pickle.load(open(results_path, "rb"))

                results_config = {key: [value] for key, value in config.items() if key in tuple(['scale'])}
                results_config.update(results)

                results_config_total = pd.concat([results_config_total, pd.DataFrame.from_dict(results_config)], axis=0)

    for pareto_front_type in ['minibatch', 'full', 'asw', 'nmi', 'ari', 'uca']:
        if pareto_front_type == 'minibatch':
            obj1 = results_config_total.loc[:, 'std_obj1_minibatch'].values.tolist()
            obj2 = results_config_total.loc[:, 'std_obj2_minibatch'].values.tolist()
            draw_plot(obj1, obj2, pareto_front_type, 'train', dir_path)
        elif pareto_front_type in ['full', 'asw', 'nmi', 'ari', 'uca']:
            for type in ['train', 'test']:
                if pareto_front_type == 'full':
                    obj1 = results_config_total.loc[:, 'std_obj1_{}set'.format(type)].values.tolist()
                    obj2 = results_config_total.loc[:, 'full_MINE_estimator_{}set'.format(type)].values.tolist()
                    draw_plot(obj1, obj2, pareto_front_type, type, dir_path)
                else:
                    obj1 = results_config_total.loc[:, '{}_be'.format(type, pareto_front_type)].values.tolist()
                    obj2 = results_config_total.loc[:, '{}_{}'.format(type, pareto_front_type)].values.tolist()
                    draw_plot(obj1, obj2, pareto_front_type, type, dir_path)
