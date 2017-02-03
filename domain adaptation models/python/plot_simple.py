#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 15:12:08 2016

@author: robbe
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt

python_path = os.path.abspath(os.getcwd())
general_path = os.path.split(python_path)[0]
input_path = os.path.join(general_path,'output')
output_path = os.path.join(general_path,'output/figures')
    
def plot_speed_compare_between_runs(test_name, num_run_start, num_run_end, input_path, output_path,f):

    ###################################################
    # line chart
    ###################################################

    fig=plt.figure()
    ax = plt.subplot(111)
    
    colors = ['blue','green','red','purple','black','pink','orange','brown','cyan','yellow']
    linestyles = ['-'] #['-','dashed', 'dashdot']
 
    min_lims = np.zeros(num_run_end-num_run_start+1)
    max_lims = np.zeros(num_run_end-num_run_start+1)
    
    #load data
    data = np.load(input_path + '/' + test_name + '_'+ str(num_run_start)+ '/results' +'.npz')
    data_np = data['train_np']
    valid_np = data['valid_np']
    test_np = data['test_np']

    #select correct part of the data
    data_steps = np.array([data_np[i][0]+data_np[i][1] for i in range(0,len(data_np))])
    data_speed = np.array([data_np[i][3] for i in range(0,len(data_np))])

    #cancel zero's in data_speed
    for i in range(len(data_speed)):
        if data_speed[i] == 0:
            data_speed[i] = data_speed[i-1]

    data_speed_mean = np.array([np.mean(data_speed[3:])])
    data_speed_std = np.array([np.std(data_speed[3:])])
    data_last_training = np.array([data_np[-1][2]])
    data_last_valid = np.array([valid_np[-1][2]])
    data_test = np.array([test_np[-1][2]])
    
    #determine axis limits
    min_lims[0] = np.percentile(data_speed,5)
    max_lims[0] = np.max(data_speed) +5
    
    #captions
    title = ''
    label = ''
    param = test_name.split('-')
    param_np = data['param_train_np']
    for i in range(0,len(param_np)):
        if param_np[i][0] in param:
            label = label +param_np[i][0] + ' = ' + param_np[i][1] + ', '
        else:
            title = title +param_np[i][0] + ' = ' + param_np[i][1] + ', '
        if (i+1) % 5 == 0:
            title = title + '\n'
    if label == '':
        label = test_name + '_' + str(num_run_start)
    else:
        while label[-1] != ',':
            label = label[:-1]
        label = label[:-1]
    while title[-1] != ',':
        title = title[:-1]
    title = title[:-1]
    labels = [label]
    
    #plotting
    ax.plot(data_steps, data_speed, color=colors[num_run_start % len(colors)], linestyle=linestyles[num_run_start % len(linestyles)], label = label)
    
    for run in range(num_run_start+1,num_run_end+1):
        data = np.load(input_path + '/' + test_name + '_'+ str(run)+ '/results' +'.npz')
        data_np = data['train_np']
        valid_np = data['valid_np']
        test_np = data['test_np']   
        
        data_steps = np.array([data_np[i][0]+data_np[i][1] for i in range(0,len(data_np))])
        data_speed = np.array([data_np[i][3] for i in range(0,len(data_np))])
	

        #cancel zero's in data_speed
        for i in range(len(data_speed)):
            if data_speed[i] == 0:
                data_speed[i] = data_speed[i-1]

        min_lims[run-num_run_start] = np.percentile(data_speed,5)
        max_lims[run-num_run_start] = np.max(data_speed) +5
    
        data_speed_mean = np.append(data_speed_mean,np.mean(data_speed[3:]))
        data_speed_std = np.append(data_speed_std,np.std(data_speed[3:]))
        data_last_training = np.append(data_last_training,data_np[-1][2])
        data_last_valid = np.append(data_last_valid,valid_np[-1][2])
        data_test = np.append(data_test,test_np[-1][2])
    
        label = ''
        param_np = data['param_train_np']
        for i in range(0,len(param_np)):
            if param_np[i][0] in param:
                    label = label +param_np[i][0] + ' = ' + param_np[i][1] + ', '
        if label == '':
            label = test_name + '_' + str(run)
        else:
            label = label[:-2]
        labels.append(label)

        ax.plot(data_steps, data_speed, color=colors[run % len(colors)], linestyle=linestyles[run % len(linestyles)], label = label)

    fig.suptitle('Compare speed plot of ' + test_name + ' from ' + str(num_run_start) + ' to ' + str(num_run_end), fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=7)
    plt.subplots_adjust(top=.855, bottom=.25)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('WPS (words per second)')
    plt.ylim([np.min(min_lims),np.max(max_lims)])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=8)
    fig.savefig(output_path + '/' + test_name + '_from_' + str(num_run_start) + '_to_' + str(num_run_end) + '_speed.png')
    plt.close()

    ###################################################
    # Bar chart
    ###################################################

 
    fig = plt.figure()
    ax = plt.subplot(111)
    
    n_groups = num_run_end + 1
    index = np.arange(n_groups)
    bar_width = 0.35

    opacity = 0.5
    error_config = {'ecolor': '0.3'}
   
    barlist = plt.bar(index + (1-bar_width)/2, data_speed_mean, bar_width,
                 alpha=opacity,
                 yerr=data_speed_std,
                 error_kw=error_config)

    for i in range(n_groups):
        barlist[i].set_color(colors[i])
	barlist[i].set_label(labels[i])

    #fig.suptitle('Compare speed plot of ' + test_name + ' from ' + str(num_run_start) + ' to ' + str(num_run_end), fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=7)
    plt.subplots_adjust(top=.855, bottom=.20)
    ax.set_xlabel('Runs')
    ax.set_ylabel('WPS (words per second)')
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=8)
    fig.savefig(output_path + '/' + test_name + '_from_' + str(num_run_start) + '_to_' + str(num_run_end) + '_speed_bar.png')
    plt.close()

    ###################################################
    # write to file
    ###################################################
    for i in range(np.size(data_speed_mean)):
#        f.write("{:<45}".format(labels[i]))
#        f.write("{:<20}".format('& ' + str(data_last_training[i])))
#        f.write("{:<20}".format('& ' + str(data_last_valid[i])))
#        f.write("{:<20}".format('& ' + str(data_test[i])))
#        f.write("{:<20}".format('& ' + str(data_speed_mean[i])))
#        f.write("{:<20}".format('& ' + str(data_speed_std[i])))
#        f.write('\n')
        
        f.write("{:<45}".format(labels[i]))
        f.write('& ')
        f.write("{:<10.4f}".format(data_last_training[i]))
        f.write('        & ')
        f.write("{:<10.4f}".format(data_last_valid[i]))
        f.write('        & ')
        f.write("{:<10.4f}".format(data_test[i]))
        f.write('        & ')
        f.write("{:<10.4f}".format(data_speed_mean[i]))
        f.write(' \\'+'\\'+' \hline')
        f.write('\n')
        
    
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
        if (i+1) % 5 == 0:
            param = param + '\n'
    if param == '':
        param = test_name + '_' + str(num_run)
    else:
        while param[-1] != ',':
            param = param[:-1]
        param = param[:-1]

    test_np = data['test_np']
    test_PPL = test_np[0][2] +0*train_steps
    
    fig = plt.figure()
    fig.suptitle('One run plot of ' + test_name + '_' + str(num_run), fontsize=14, fontweight='bold')

    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.86)
    ax.set_title(param, fontsize=7)

    ax.set_xlabel('Steps')
    ax.set_ylabel('Perplexity')

    min_lims = np.zeros(3)
    max_lims = np.zeros(3)
    min_lims[0] = np.min(train_PPL)-5
    min_lims[1] = np.min(valid_PPL)-5	
    min_lims[2] = np.min(test_PPL)-5
    max_lims[0] = np.percentile(train_PPL,85)
    max_lims[1] = np.percentile(valid_PPL,85)	
    max_lims[2] = np.percentile(test_PPL,85)
    
    plt.ylim([np.min(min_lims),np.max(max_lims)])

    ax.plot(train_steps, train_PPL, color="blue", linewidth=1.0, linestyle="-", label = 'train')
    ax.plot(valid_steps, valid_PPL, color="green", linewidth=1.0, linestyle="-", label = 'validation')
    ax.plot(train_steps, test_PPL, color="red", linewidth=1.0, linestyle="-", label = 'test')
    ax.legend(loc='upper right', fontsize=10)
    fig.savefig(output_path + '/' + test_name + '_' + str(num_run)+ '.png')
    plt.close()
    
def plot_heat_table(test_name,num_run_start, num_run_end, input_paht, output_path):
    param = test_name.split('-')
    if len(param) == 1:
        fig=plt.figure(figsize=(20, 11))
        fig.suptitle('Heatmap ' + test_name, fontsize=14, fontweight='bold')
        ax = fig.add_subplot(111, frameon=False, xticks = [], yticks = [])
        
        validPPL = np.zeros((1,num_run_end-num_run_start+1))
        testPPL = np.zeros((1,num_run_end-num_run_start+1))
        
        data = np.load(input_path + '/' + test_name + '_'+ str(num_run_start)+ '/results' +'.npz')
        validPPL[0][0] = data['valid_np'][-1][2]
        testPPL[0][0] = data['test_np'][-1][2]
        
        label = ''
        title = ''
        param_np = data['param_train_np']
        for i in range(0,len(param_np)):
            if param_np[i][0] in param:
                label = label +param_np[i][0] + ' = ' + param_np[i][1] + ', '
            else:
                title = title +param_np[i][0] + ' = ' + param_np[i][1] + ', '
            if (i+1) % 5 == 0:
                title = title + '\n'
        if label == '':
            label = test_name + '_' + str(num_run_start)
        else:
            while label[-1] != ',':
                label = label[:-1]
            label = label[:-1]
        while title[-1] != ',':
            title = title[:-1]
        title = title[:-1]
        coll = [label]
        rowl = ['']
        
        if label == []:
            return
        
        for run in range(num_run_start+1,num_run_end+1):
            data = np.load(input_path + '/' + test_name + '_'+ str(run)+ '/results' +'.npz')
            validPPL[0][run-num_run_start]  = data['valid_np'][-1][2]
            testPPL[0][run-num_run_start] = data['test_np'][-1][2]

            label = ''
            param_np = data['param_train_np']
            for i in range(0,len(param_np)):
                if param_np[i][0] in param:
                    label = label +param_np[i][0] + ' = ' + param_np[i][1]
            coll.append(label)
        
        tekst = [[[]for i in range(len(validPPL[0]))] for j in range(len(validPPL))]
        for i in range(len(validPPL)):
            for j in range(len(validPPL[0])):
                tekst[i][j] = str("%.2f" % validPPL[i][j]) + ' / ' + str("%.2f" % testPPL[i][j])
        img = plt.imshow(testPPL, cmap="OrRd", vmin = 80, vmax = 130)
        #plt.colorbar()
        img.set_visible(False)
        tb = plt.table(cellText = tekst, cellLoc='center', rowLabels = rowl, colLabels = coll, loc = 'center', cellColours = img.to_rgba(testPPL))
        tb.scale(1, 5)
        ax.set_title(title, fontsize=13)
        fig.savefig(output_path + '/heatmap_' + test_name + '.png')
        plt.close()
    elif len(param) == 2:
        fig=plt.figure(figsize=(20, 11))
        fig.suptitle('Heatmap ' + test_name, fontsize=14, fontweight='bold')
        ax = fig.add_subplot(111, frameon=False, xticks = [], yticks = [])
        
        param1 = param[0]
        param2 = param[1]
        coll = []
        rowl = []

        for run in range(num_run_start,num_run_end+1):
            data = np.load(input_path + '/' + test_name + '_'+ str(run)+ '/results' +'.npz')
            param_np = data['param_train_np']
            for i in range(0,len(param_np)):
                if param_np[i][0] == param1:
                    label = param_np[i][0] + ' = ' + param_np[i][1]
                    if label not in rowl:
                        rowl.append(label)
                elif param_np[i][0] == param2:
                    label = param_np[i][0] + ' = ' + param_np[i][1]
                    if label not in coll:
                        coll.append(label)
        if rowl == [] or coll == []:
            return 
        
        title = ''
        for i in range(0,len(param_np)):
            if param_np[i][0] not in param:
                title = title +param_np[i][0] + ' = ' + param_np[i][1] + ', '
            if (i+1) % 5 == 0:
                title = title + '\n'
        while title[-1] != ',':
            title = title[:-1]
        title = title[:-1]
            
        validPPL = np.zeros((len(rowl),len(coll)))
        testPPL = np.zeros((len(rowl),len(coll)))
        
        for run in range(num_run_start,num_run_end+1):
            data = np.load(input_path + '/' + test_name + '_'+ str(run)+ '/results' +'.npz')
            param_np = data['param_train_np']
            for i in range(0,len(param_np)):
                if param_np[i][0] == param1:
                    label = param_np[i][0] + ' = ' + param_np[i][1]
                    row = rowl.index(label)
                elif param_np[i][0] == param2:
                    label = param_np[i][0] + ' = ' + param_np[i][1]
                    col = coll.index(label)

            validPPL[row][col]  = data['valid_np'][-1][2]
            testPPL[row][col] = data['test_np'][-1][2]
        
        
        tekst = [[[]for i in range(len(validPPL[0]))] for j in range(len(validPPL))]
        for i in range(len(validPPL)):
            for j in range(len(validPPL[0])):
                tekst[i][j] = str("%.2f" % validPPL[i][j]) + ' / ' + str("%.2f" % testPPL[i][j])
        img = plt.imshow(testPPL, cmap="OrRd", vmin = 80, vmax = 130)
        #plt.colorbar()
        img.set_visible(False)
        tb = plt.table(cellText = tekst, cellLoc='center', rowLabels = rowl, colLabels = coll, loc = 'center', cellColours = img.to_rgba(testPPL))
        tb.scale(1, 5)
        ax.set_title(title, fontsize=13)
        fig.savefig(output_path + '/heatmap_' + test_name + '.png')
        plt.close()
    else:
        return
            
    
    
def plot_compare_between_runs(test_name, num_run_start, num_run_end, train_valid_test, input_path, output_path):
    train_valid_test_str = train_valid_test + '_np'
    fig = plt.figure()
    #fig.suptitle('Compare '+train_valid_test+ ' plot of ' + test_name + ' from ' + str(num_run_start) + ' to ' + str(num_run_end), fontsize=14, fontweight='bold')
    
    colors = ['blue','green','red','purple','black','pink','orange','cyan','yellow']
    markers = [ '+' , ',' , '.' , '1' , '2' , '3' , '4' , None]
    linestyles = ['-'] #['-','dashed', 'dashdot']
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.86)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Perplexity')
    
    min_lims = np.zeros(num_run_end-num_run_start+1)
    max_lims = np.zeros(num_run_end-num_run_start+1)
    
    param = test_name.split('-')
    
    data = np.load(input_path + '/' + test_name + '_'+ str(num_run_start)+ '/results' +'.npz')
    data_np = data[train_valid_test_str]
    data_steps = np.array([data_np[i][0]+data_np[i][1] for i in range(0,len(data_np))])
    data_PPL = np.array([data_np[i][2] for i in range(0,len(data_np))])
    
    min_lims[0] = np.min(data_PPL)-5
    max_lims[0] = np.percentile(data_PPL,85)
        
    title = ''
    label = ''
    param = test_name.split('-')
    param_np = data['param_train_np']
    for i in range(0,len(param_np)):
        if param_np[i][0] in param:
            label = label +param_np[i][0] + ' = ' + param_np[i][1] + ', '
        else:
            title = title +param_np[i][0] + ' = ' + param_np[i][1] + ', '
        if (i+1) % 5 == 0:
            title = title + '\n'
    if label == '':
        label = test_name + '_' + str(num_run_start)
    else:
        while label[-1] != ',':
            label = label[:-1]
        label = label[:-1]
    while title[-1] != ',':
        title = title[:-1]
    title = title[:-1]
    labels = [label]
    
    if train_valid_test == 'test':
        ax.plot(data_steps, data_PPL, color=colors[num_run_start % len(colors)], marker='+', label = label)
    else:
        ax.plot(data_steps, data_PPL, color=colors[num_run_start % len(colors)], linestyle=linestyles[num_run_start % len(linestyles)], marker = markers[num_run_start % len(markers)], label = label)
    ax.set_title(title, fontsize=7)
    
    for run in range(num_run_start+1,num_run_end+1):
        data = np.load(input_path + '/' + test_name + '_'+ str(run)+ '/results' +'.npz')
        data_np = data[train_valid_test_str]
        
        data_steps = np.array([data_np[i][0]+data_np[i][1] for i in range(0,len(data_np))])
        data_PPL = np.array([data_np[i][2] for i in range(0,len(data_np))])
        min_lims[run-num_run_start] = np.min(data_PPL)-5
        max_lims[run-num_run_start] = np.percentile(data_PPL,85)

        label = ''
        param_np = data['param_train_np']
        for i in range(0,len(param_np)):
            if param_np[i][0] in param:
                    label = label +param_np[i][0] + ' = ' + param_np[i][1] + ', '
        if label == '':
            label = test_name + '_' + str(run)
        else:
            label = label[:-2]
        labels.append(label)

        if train_valid_test == 'test':
            ax.plot(data_steps, data_PPL, color=colors[run % len(colors)], marker='+', label = label)    
        else:
            ax.plot(data_steps, data_PPL, color=colors[run % len(colors)], marker = markers[run % len(markers)], linestyle=linestyles[run % len(linestyles)], label = label)
    
    #plt.ylim([85,250])
    plt.ylim([np.min(min_lims),np.max(max_lims)])
    ax.legend(loc='upper right', fontsize=8)
    fig.savefig(output_path + '/' + test_name + '_from_' + str(num_run_start) + '_to_' + str(num_run_end) + '_' +train_valid_test+ '.png')
    plt.close()

def plot_compare_between_runs_summary(test_name, num_run_start, num_run_end, input_path, output_path):
    ###################################################
    # line chart
    ###################################################

    fig=plt.figure()
    ax = plt.subplot(111)
    
    colors = ['blue','green','red','purple','black','pink','orange','brown','cyan','yellow']
    markers = [ '+' , ',' , '.' , '1' , '2' , '3' , '4' ,None]
    linestyles = ['-'] #['-','dashed', 'dashdot']
 
    min_lims = np.zeros(num_run_end-num_run_start+1)
    max_lims = np.zeros(num_run_end-num_run_start+1)
    
    

    #load data
    data = np.load(input_path + '/' + test_name + '_'+ str(num_run_start)+ '/results' +'.npz')
    valid_np = data['valid_np']
    test_np = data['test_np']

    data_steps = np.array([valid_np[i][0]+valid_np[i][1] for i in range(0,len(valid_np))])
    valid_PPL = np.array([valid_np[i][2] for i in range(0,len(valid_np))])
    test_PPL = test_np[0][2] +0*data_steps

    min_lims[0] = np.min(np.concatenate((valid_PPL,test_PPL),axis=0))-5
    max_lims[0] = np.percentile(np.concatenate((valid_PPL,test_PPL),axis=0),95)
        
    title = ''
    label = ''
    param = test_name.split('-')
    param_np = data['param_train_np']
    for i in range(0,len(param_np)):
        if param_np[i][0] in param:
            label = label +param_np[i][0] + ' = ' + param_np[i][1] + ', '
        else:
            title = title +param_np[i][0] + ' = ' + param_np[i][1] + ', '
        if (i+1) % 5 == 0:
            title = title + '\n'
    if label == '':
        label = test_name + '_' + str(num_run_start)
    else:
        while label[-1] != ',':
            label = label[:-1]
        label = label[:-1]
    while title[-1] != ',':
        title = title[:-1]
    title = title[:-1]
    labels = [label]
    
    #ax.plot(data_steps, test_PPL, color=colors[num_run_start % len(colors)], marker = markers[num_run_start % len(markers)], linestyle='--')
    ax.plot(data_steps, valid_PPL, color=colors[num_run_start % len(colors)], marker = markers[num_run_start % len(markers)], linestyle='-', label = label)
    
    for run in range(num_run_start+1,num_run_end+1):
        data = np.load(input_path + '/' + test_name + '_'+ str(run)+ '/results' +'.npz')
        valid_np = data['valid_np']
        test_np = data['test_np']
        
        data_steps = np.array([valid_np[i][0]+valid_np[i][1] for i in range(0,len(valid_np))])
        valid_PPL = np.array([valid_np[i][2] for i in range(0,len(valid_np))])
        test_PPL = test_np[0][2] +0*data_steps

        min_lims[run-num_run_start] = np.min(np.concatenate((valid_PPL,test_PPL),axis=0))-5
        max_lims[run-num_run_start] = np.percentile(np.concatenate((valid_PPL,test_PPL),axis=0),95)

        label = ''
        param_np = data['param_train_np']
        for i in range(0,len(param_np)):
            if param_np[i][0] in param:
                    label = label +param_np[i][0] + ' = ' + param_np[i][1] + ', '
        if label == '':
            label = test_name + '_' + str(run)
        else:
            label = label[:-2]
        labels.append(label)
        
        #ax.plot(data_steps, test_PPL, color=colors[run % len(colors)], marker = markers[run % len(markers)], linestyle='--')    
        ax.plot(data_steps, valid_PPL, color=colors[run % len(colors)], marker = markers[run % len(markers)], linestyle='-', label = label)

    #fig.suptitle('Compare valid en test plot of ' + test_name + ' from ' + str(num_run_start) + ' to ' + str(num_run_end), fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=7)
    plt.subplots_adjust(top=.855, bottom=.25)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('PPL')
    plt.ylim([np.min(min_lims),np.max(max_lims)])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=8)
    fig.savefig(output_path + '/' + test_name + '_from_' + str(num_run_start) + '_to_' + str(num_run_end) + '_val_test.png')
    plt.close()

def plot_certain_files(dirs, input_path, output_path, name):
    fig=plt.figure()
    ax = plt.subplot(111)
    
    colors = ['blue','green','red','purple','black','pink','orange','brown','cyan','yellow']
    markers = [ '+' , ',' , '.' , '1' , '2' , '3' , '4' ,None]
    linestyles = ['-'] #['-','dashed', 'dashdot']
 
    min_lims = np.zeros(len(dirs))
    max_lims = np.zeros(len(dirs))
    
    

    #load data
    data = np.load(input_path + '/' +dirs[0] + '/results' +'.npz')
    valid_np = data['valid_np']
    test_np = data['test_np']

    data_steps = np.array([valid_np[i][0]+valid_np[i][1] for i in range(0,len(valid_np))])
    valid_PPL = np.array([valid_np[i][2] for i in range(0,len(valid_np))])
    test_PPL = test_np[0][2] +0*data_steps

    min_lims[0] = np.min(np.concatenate((valid_PPL,test_PPL),axis=0))-5
    max_lims[0] = np.percentile(np.concatenate((valid_PPL,test_PPL),axis=0),95)
        
    title = name
    label = dirs[0][0:-2]
    labels = [label]
    
    ax.plot(data_steps, test_PPL, color=colors[0 % len(colors)], marker = markers[0 % len(markers)], linestyle='--')
    ax.plot(data_steps, valid_PPL, color=colors[0 % len(colors)], marker = markers[0 % len(markers)], linestyle='-', label = label)
    
    for run in range(1,len(dirs)):
        data = np.load(input_path + '/' +dirs[run]+ '/results' +'.npz')
        valid_np = data['valid_np']
        test_np = data['test_np']
        
        data_steps = np.array([valid_np[i][0]+valid_np[i][1] for i in range(0,len(valid_np))])
        valid_PPL = np.array([valid_np[i][2] for i in range(0,len(valid_np))])
        test_PPL = test_np[0][2] +0*data_steps

        min_lims[run] = np.min(np.concatenate((valid_PPL,test_PPL),axis=0))-5
        max_lims[run] = np.percentile(np.concatenate((valid_PPL,test_PPL),axis=0),95)

        label = dirs[run][0:-2]
        labels.append(label)
        
        ax.plot(data_steps, test_PPL, color=colors[run % len(colors)], marker = markers[run % len(markers)], linestyle='--')    
        ax.plot(data_steps, valid_PPL, color=colors[run % len(colors)], marker = markers[run % len(markers)], linestyle='-', label = label)

    #fig.suptitle('Compare valid en test plot of ' + test_name + ' from ' + str(num_run_start) + ' to ' + str(num_run_end), fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=7)
    plt.subplots_adjust(top=.855, bottom=.25)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('PPL')
    plt.ylim([np.min(min_lims),np.max(max_lims)])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=8)
    fig.savefig(output_path + '/' + name +'.png')
    plt.close()
    
    

def plot_everything(input_path, output_path):
    python_path = os.path.abspath(os.getcwd())
    general_path = os.path.split(python_path)[0]
    
    input_path = os.path.join(general_path,'output')
    output_path = os.path.join(general_path,'output/figures')
    dirs = os.listdir(input_path)
    if 'figures' in dirs:
        dirs.remove('figures')
    for i in xrange(len(dirs)):
        dirs[i] = ''.join([j for j in dirs[i] if not j.isdigit()])
        dirs[i] = dirs[i][:-1]

    dirsset = list(set(dirs))
    dirsdic = dict()    
    for elem in dirsset:
        dirsdic[elem] = dirs.count(elem)
     
    f = open(output_path + '/summary_resulst.txt', 'w')
    f.write("{:<45}".format('TEST_NAME'))
    f.write("{:<20}".format('LAST_TRAINING_PPL'))
    f.write("{:<20}".format('LAST_VALID_PPL'))  
    f.write("{:<20}".format('TEST_PPL'))  
    f.write("{:<20}".format('AVERAGE_SPEED [WPS]'))
    f.write('\n')



    #printing
    for test_name in dirsdic.keys():
        plot_compare_between_runs(test_name, 0, dirsdic[test_name]-1, 'train', input_path, output_path)
        plot_compare_between_runs(test_name, 0, dirsdic[test_name]-1, 'test', input_path, output_path)
        plot_compare_between_runs(test_name, 0, dirsdic[test_name]-1, 'valid', input_path, output_path)
        plot_heat_table(test_name,0, dirsdic[test_name]-1, input_path, output_path)
        plot_compare_between_runs_summary(test_name, 0, dirsdic[test_name]-1, input_path, output_path)
        plot_speed_compare_between_runs(test_name, 0, dirsdic[test_name]-1, input_path, output_path,f)
        for i in xrange(dirsdic[test_name]):
            plot_PPL_one_run(test_name, i, input_path, output_path)
    
    dirs = os.listdir(input_path)
    if 'figures' in dirs:
        dirs.remove('figures')
    plot_certain_files(dirs, input_path, output_path, 'comparing of all methods')
    f.close() 

def main():
    plot_everything(input_path, output_path)

if __name__ == "__main__":
    main()
