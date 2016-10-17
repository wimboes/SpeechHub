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

# parser = argparse.ArgumentParser(description='Print comparation of train-, valid or test-PPL of certain runs')
# parser.add_argument("test_name", help="give the name of the test you wan't to show")
# parser.add_argument("num_run_start", help="give the startnumber of the tests you wan't to compare",
#                     type=int)
# parser.add_argument("num_run_end", help="give the endnumber of the tests you wan't to compare",
#                     type=int)
# args = parser.parse_args()

def plot_speed_compare_between_runs(test_name, num_run_start, num_run_end, input_path, output_path):

    fig = plt.figure()
    fig.suptitle('Compare speed plot of ' + test_name + ' from ' + str(num_run_start) + ' to ' + str(num_run_end), fontsize=14, fontweight='bold')
    
    colors = ['blue','green','red','purple','black','pink','orange','brown','yellow']
    linestyles = ['--'] #['-','dashed', 'dashdot']
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.86)
    ax.set_xlabel('Steps')
    ax.set_ylabel('WPS (words per second)')
    min_lims = np.zeros(num_run_end-num_run_start+1)
    max_lims = np.zeros(num_run_end-num_run_start+1)
    
    
    param = test_name.split('-')
    
    data = np.load(input_path + '/' + test_name + '_'+ str(num_run_start)+ '/results' +'.npz')
    data_np = data['train_np']
    data_steps = np.array([data_np[i][0]+data_np[i][1] for i in range(0,len(data_np))])
    data_speed = np.array([data_np[i][3] for i in range(0,len(data_np))])

    #cancel zero's in data_speed
    for i in range(len(data_speed)):
        if data_speed[i] == 0:
            data_speed[i] = data_speed[i-1]

    min_lims[0] = np.percentile(data_speed,5)
    max_lims[0] = np.max(data_speed) +5
    
    label = ''
    title = ''
    param_np = data['param_train_np']
    for i in range(0,len(param_np)):
        if param_np[i][0] in param:
            label = label +param_np[i][0] + ' = ' + param_np[i][1] + ', '
        else:
            title = title +param_np[i][0] + ' = ' + param_np[i][1] + ', '
        if (i+1) % 4 == 0:
            title = title + '\n'
    while label[-1] != ',':
	label = label[:-1]
    label = label[:-1]
    while title[-1] != ',':
	title = title[:-1]
    title = title[:-1]
    
    ax.plot(data_steps, data_speed, color=colors[num_run_start % len(colors)], linestyle=linestyles[num_run_start % len(linestyles)], label = label)
    ax.set_title(title, fontsize=7)
    
    for run in range(num_run_start+1,num_run_end+1):
        data = np.load(input_path + '/' + test_name + '_'+ str(run)+ '/results' +'.npz')
        data_np = data['train_np']
        
        data_steps = np.array([data_np[i][0]+data_np[i][1] for i in range(0,len(data_np))])
        data_speed = np.array([data_np[i][3] for i in range(0,len(data_np))])

	#cancel zero's in data_speed
    	for i in range(len(data_speed)):
        	if data_speed[i] == 0:
            		data_speed[i] = data_speed[i-1]
        min_lims[run-num_run_start] = np.percentile(data_speed,5)
        max_lims[run-num_run_start] = np.max(data_speed) +5

        label = ''
        param_np = data['param_train_np']
        for i in range(0,len(param_np)):
            if param_np[i][0] in param:
                label = label +param_np[i][0] + ' = ' + param_np[i][1] + ', '
        label = label[:-2]

        ax.plot(data_steps, data_speed, color=colors[run % len(colors)], linestyle=linestyles[run % len(linestyles)], label = label)
    plt.ylim([np.min(min_lims),np.max(max_lims)])
    ax.legend(loc='upper right', fontsize=8)
    fig.savefig(output_path + '/' + test_name + '_from_' + str(num_run_start) + '_to_' + str(num_run_end) + '_speed.png')
    plt.close()

def main():
    plot_speed_compare_between_runs(args.test_name, args.num_run_start, args.num_run_end, input_path, output_path)

if __name__ == "__main__":
    main()


