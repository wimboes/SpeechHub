# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 12:46:49 2017

@author: robbe
!First remove old batch-files!!
"""
import os
import reader

python_path = os.path.abspath(os.getcwd())
general_path = os.path.split(python_path)[0]
path = os.path.join(os.path.split(os.path.split(python_path)[0])[0],'input')

a = reader.ds_data_sentence(50,path,'ds.train.txt')
del a
a = reader.ds_data_sentence(50,path,'ds.valid.txt')
del a
a = reader.ds_data_sentence(1,path,'ds.valid.txt')
del a
a = reader.ds_data_sentence(50,path,'ds.test.txt')
del a
a = reader.ds_data_sentence(50,path,'ds.testshort.txt')
del a
a = reader.ds_data_sentence(1,path,'ds.test.txt')
del a
a = reader.ds_data_sentence(1,path,'ds.testshort.txt')
del a

a = reader.ds_data_continuous(50,50,path,'ds.train.txt')
del a
a = reader.ds_data_continuous(50,50,path,'ds.valid.txt')
del a
a = reader.ds_data_continuous(1,1,path,'ds.valid.txt')
del a
a = reader.ds_data_continuous(50,50,path,'ds.test.txt')
del a
a = reader.ds_data_continuous(1,1,path,'ds.test.txt')
del a
a = reader.ds_data_continous(1,1,path,'ds.testshort.txt')
del a