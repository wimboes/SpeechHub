# authors : Wim Boes & Robbe Van Rompaey
# date: 17-10-2016 

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

parser = argparse.ArgumentParser(description='Print speed of the training of a certain run')
parser.add_argument("test_name", help="give the name of the test you wan't to show")
parser.add_argument("num_run", help="give the number of the test you wan't to show",
                    type=int)
args = parser.parse_args()

def plot_speed_one_run(test_name, num_run, input_path, output_path):
    data = np.load(input_path + '/' + test_name + '_'+ str(num_run)+ '/results' +'.npz')
    
    train_np = data['train_np']
    train_steps = np.array([train_np[i][0]+train_np[i][1] for i in range(0,len(train_np))])
    train_speed = np.array([train_np[i][3] for i in range(0,len(train_np))])
    
    param_train_np = data['param_train_np']
    param = ''
    
    for i in range(0,len(param_train_np)):
        param = param + param_train_np[i][0] + ' = ' + param_train_np[i][1] + ', '
        if (i+1) % 4 == 0:
            param = param + '\n'
    while param[-1] != ',':
	param = param[:-1]
    param = param[:-1]
    
    fig = plt.figure()
    fig.suptitle('One run speed plot of ' + test_name + '_' + str(num_run), fontsize=14, fontweight='bold')

    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.86)
    ax.set_title(param, fontsize=7)

    ax.set_xlabel('Steps')
    ax.set_ylabel('WPS (words per second)')
    #plt.ylim([60,1000])

    ax.plot(train_steps, train_PPL, color="blue", linewidth=1.0, linestyle="-")
    fig.savefig(output_path + '/' + test_name + '_' + str(num_run)+ '_speed' + '.png')

def main():
    plot_speed_one_run(args.test_name, args.num_run, input_path, output_path)

if __name__ == "__main__":
    main()

