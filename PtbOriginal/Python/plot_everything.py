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
from plot_PPL_compare_between_runs import plot_compare_between_runs, plot_compare_between_runs_summary
from plot_PPL_one_run import plot_PPL_one_run
from plot_speed_compare_between_runs import plot_speed_compare_between_runs

python_path = os.path.abspath(os.getcwd())
general_path = os.path.split(python_path)[0]
input_path = os.path.join(general_path,'Output')
output_path = os.path.join(general_path,'Output/Figures')

#parser = argparse.ArgumentParser(description='Print train-, valid and test-PPL of a certain run')

def plot_everything(input_path, output_path):
    dirs = os.listdir(input_path)
    if 'Figures' in dirs:
	dirs.remove('Figures')
    for i in xrange(len(dirs)):
	dirs[i] = ''.join([j for j in dirs[i] if not j.isdigit()])
	dirs[i] = dirs[i][:-1]
    dirsset = list(set(dirs))
    dirsdic = dict()    
    for elem in dirsset:
    	dirsdic[elem] = dirs.count(elem)

    #printing
    for test_name in dirsdic.keys():
	plot_compare_between_runs(test_name, 0, dirsdic[test_name]-1, 'train', input_path, output_path)
	plot_compare_between_runs(test_name, 0, dirsdic[test_name]-1, 'test', input_path, output_path)
	plot_compare_between_runs(test_name, 0, dirsdic[test_name]-1, 'valid', input_path, output_path)
	plot_compare_between_runs_summary(test_name, 0, dirsdic[test_name]-1, input_path, output_path)
    	plot_speed_compare_between_runs(test_name, 0, dirsdic[test_name]-1, input_path, output_path)
    	for i in xrange(dirsdic[test_name]):
        	plot_PPL_one_run(test_name, i, input_path, output_path)
    

def main():
    plot_everything(input_path, output_path)

if __name__ == "__main__":
    main()


