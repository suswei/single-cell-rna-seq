import os
import itertools
import numpy as np
import pandas as pd
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

def pareto_front(hyperparameter_config, save_path: str='./result/pareto_front_scVI_MINE/muris_tabula/batch/',
                dataset_name: str= 'muris_tabula', confounder: str='batch'):

    clustermetric = pd.DataFrame(columns=['Label', 'asw', 'nmi', 'ari', 'uca', 'be', 'std_penalty', 'std_ELBO', 'penalty_fully'])

    rep_scale = pd.DataFrame(columns=['rep','scale'])
    for i in range(rep_number):
        for j in range(10):
            file_dir = input_dir + 'rep%s/'%(i) + 'muris_tabula_batch_MIScale%s_ClusterMetric.csv'%(j)
            if os.path.isfile(file_dir):
                clustermetric_onerep_onescale = pd.read_csv(file_dir)
                rep_scale_one = pd.DataFrame.from_dict({'rep':[i,i], 'scale':[j,j]})
                clustermetric = pd.concat([clustermetric, clustermetric_onerep_onescale], axis=0)
                rep_scale = pd.concat([rep_scale_one, rep_scale], axis=0)
    clustermetric = pd.concat([rep_scale, clustermetric], axis=1)
    clustermetric_half = clustermetric[clustermetric['Label'].str.match('muris_tabula_batch_MIScale.*._VaeMI_' + Label)]
    std_penalty = clustermetric_half.loc[:,'std_penalty'].values.tolist()
    std_ELBO = clustermetric_half.loc[:,'std_ELBO'].values.tolist()
    penalty_full = clustermetric_half.loc[:,'penalty_fully'].values.tolist()
    inputPoints1 = [[std_ELBO[k],std_penalty[k]] for k in range(len(std_penalty))]
    inputPoints2 = [[std_ELBO[k],penalty_full[k]] for k in range(len(penalty_full))]
    paretoPoints1, dominatedPoints1 = simple_cull(inputPoints1, dominates)
    paretoPoints2, dominatedPoints2 = simple_cull(inputPoints2, dominates)

    fig = plt.figure()
    dp = np.array(list(dominatedPoints1))
    pp = np.array(list(paretoPoints1))
    plt.scatter(dp[:,0],dp[:,1])
    plt.scatter(pp[:,0],pp[:,1],color='red')
    plt.title('%s'%(Label), fontsize=18)
    plt.xlabel('std_ELBO', fontsize=16)
    plt.ylabel('std_penalty', fontsize=16)
    fig.savefig(output_dir + '%s_%s_%s_pareto_front.png' % (dataset_name, nuisance_variable, Label))
    plt.close(fig)

    fig = plt.figure()
    dp = np.array(list(dominatedPoints2))
    pp = np.array(list(paretoPoints2))
    plt.scatter(dp[:, 0], dp[:, 1])
    plt.scatter(pp[:, 0], pp[:, 1], color='red')
    plt.title('%s' % (Label), fontsize=18)
    plt.xlabel('std_ELBO', fontsize=16)
    plt.ylabel('penalty_full', fontsize=16)
    fig.savefig(output_dir + '%s_%s_%s_penalty_full_pareto_front.png' % (dataset_name, nuisance_variable, Label))
    plt.close(fig)
