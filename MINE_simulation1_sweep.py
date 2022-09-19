import sys
import os
import itertools

def main(taskid):

    taskid = int(taskid[0])

    hyperparameter_config = {
        'case_idx': list(range(7)),
        'category_num': [10],
        'gaussian_dim': [2],
        'samplesize': [25600],
        'n_hidden_node': [128],
        'n_hidden_layer': [10],
        'activation_fun': ['ELU'],
        'batch_size': [512],
        'epochs': [400],
        'lr': [5e-5],
        'MC': list(range(20))
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    temp = hyperparameter_experiments[taskid]

    os.system("python3 MINE_simulation1_main.py --unbiased_loss --taskid %s --case_idx %s --category_num %s --gaussian_dim %s "
              "--samplesize %s --n_hidden_node %s --n_hidden_layer %s --activation_fun %s --batch_size %s "
              "--epochs %s --lr %s --MC %s"
              %(taskid, temp['case_idx'], temp['category_num'],temp['gaussian_dim'], temp['samplesize'],
                temp['n_hidden_node'], temp['n_hidden_layer'], temp['activation_fun'], temp['batch_size'],
                temp['epochs'], temp['lr'], temp['MC'])
              )

if __name__ == "__main__":
    main(sys.argv[1:])
