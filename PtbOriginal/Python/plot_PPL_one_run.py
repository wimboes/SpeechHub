# authors : Wim Boes & Robbe Van Rompaey
# date: 12-10-2016 

# imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

python_path = os.path.abspath(os.getcwd())
general_path = os.path.split(python_path)[0]
input_path = os.path.join(general_path,'Output')
output_path = os.path.join(general_path,'Output/Figures')

parser = argparse.ArgumentParser(description='Print train-, valid and test-PPL of a certain run')
parser.add_argument("test_name", help="give the name of the test you wan't to show")
parser.add_argument("num_run", help="give the number of the test you wan't to show",
                    type=int)
args = parser.parse_args()

def plot_PPL_one_run(test_name, num_run, input_path, output_path):
    data = np.load(input_path + '/' + test_name + '_'+ str(num_run)+ '/results' +'.npz')
    
    train_np = data['train_np']
    train_steps = np.array([train_np[i][0]+train_np[i][1] for i in range(0,len(train_np))])
    train_PPL = np.array([train_np[i][2] for i in range(0,len(train_np))])
    
    valid_np = data['valid_np']
    valid_steps = np.array([valid_np[i][0]+valid_np[i][1] for i in range(0,len(valid_np))])
    valid_PPL = np.array([valid_np[i][2] for i in range(0,len(valid_np))])
    
    param_train_np = data['param_train_np']
    param = ''
    
    for i in range(0,len(param_train_np)):
        param = param + param_train_np[i][0] + ' = ' + param_train_np[i][1] + ', '
        if (i+1) % 4 == 0:
            param = param + '\n'
    param = param[:-3]
    param = param + '\n'
    test_np = data['test_np']
    #test_np = np.array([[0,0,500,0]])
    test_PPL = test_np[0][2] +0*train_steps
    
    fig = plt.figure()
    fig.suptitle('One run plot of ' + test_name + '_' + str(num_run), fontsize=14, fontweight='bold')

    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    ax.set_title(param, fontsize=7)

    ax.set_xlabel('Steps')
    ax.set_ylabel('Perplexity')

    ax.plot(train_steps, train_PPL, color="blue", linewidth=1.0, linestyle="-", label = 'train')
    ax.plot(valid_steps, valid_PPL, color="green", linewidth=1.0, linestyle="-", label = 'validation')
    ax.plot(train_steps, test_PPL, color="red", linewidth=1.0, linestyle="-", label = 'test')
    ax.legend(loc='upper right')
    fig.savefig(output_path + '/' + test_name + '_' + str(num_run)+ '.png')

def main():
    plot_PPL_one_run(args.test_name, args.num_run, input_path, output_path)

if __name__ == "__main__":
    main()


