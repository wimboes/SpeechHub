# imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np

import reader
import fnmatch
import shutil
from gensim import corpora, models


python_path = os.path.abspath(os.getcwd())
general_path = os.path.split(python_path)[0]
input_path = os.path.join(os.path.split(os.path.split(python_path)[0])[0],'input_n_best/original_n-best')
data_path = os.path.join(os.path.split(os.path.split(python_path)[0])[0],'input')
output_path = os.path.join(general_path,'output')

# set data and save path


name = "n-best-4gram7"

save_path = output_path

def find_n_best_lists(n_best):
    n_best_files = os.listdir(n_best)
    n_best_files.sort()
    fv_files = []
    fv_files_amount = []
    for file in n_best_files:
        if not file.split('.')[0] in fv_files:
	   fv_files.append(file.split('.')[0])
	   fv_files_amount.append(int(file.split('.')[2]))
        else:
	   if fv_files_amount[-1] < int(file.split('.')[2]):
	       fv_files_amount[-1] = int(file.split('.')[2])
    
    return fv_files, fv_files_amount

def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))

print('N-best list rescoring started')

dict_path = os.path.join(data_path, "dictionary.ds")
dictionary = corpora.Dictionary.load(dict_path)
word_to_id = dict()
for (wordid,word) in dictionary.iteritems():
    word_to_id[word] = wordid   
word_to_id_set = set(word_to_id.keys())
    
output_dir = os.path.join(save_path, name)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
else:
    remove(output_dir)
    os.mkdir(output_dir)
        
fv_files, fv_files_amount = find_n_best_lists(input_path)
for i in range(199,0,-1):
    print(fv_files[i])
    for j in range(fv_files_amount[i]+1):
        name_file = fv_files[i] + '.0.' + str(j) + '.10000-best.txt'
        output_file = os.path.join(output_dir, name_file)
        if not os.path.exists(os.path.join(output_dir,'hypothesis_files' + str(j))):
            os.mkdir(os.path.join(output_dir,'hypothesis_files' + str(j)))
        else:
            remove(os.path.join(output_dir,'hypothesis_files' + str(j)))
            os.mkdir(os.path.join(output_dir,'hypothesis_files' + str(j)))

        input_file = os.path.join(input_path, name_file)
        sentences = []
        #read all sentence hypotheses	
        with open(input_file,'r') as f:
            for k in range(100): 
                line = f.readline().decode('utf-8')
                if line == '':
                    break
                sentences.append(line)	
	
        #make sentence files that later needs to be grades
        accoustic_score = [float(sentence.split()[0]) for sentence in sentences]	
            
        for k in range(len(sentences)):
            sen = sentences[k].split()
            new_sen = []
            for l in range(len(sen)): 
                if sen[l] in word_to_id_set:
                    new_sen.append(sen[l].encode('utf-8'))
            new_sen = ' '.join(new_sen)
                
            with open(os.path.join(output_dir,'hypothesis_files' + str(j) + '/hypothesis'+str(k)+'.txt'),'w') as g:
                g.write(new_sen) 	
            
        #compute perplexity
        ppl = []
        for k in range(len(sentences)):
            eval_name = os.path.join(output_dir,'hypothesis_files' + str(j),'hypothesis'+str(k)+'.txt')
            
	    temp = os.popen("ngram -ppl " +  eval_name + " -order 4 -lm "+ data_path + "/4gram.mod")
            a = temp.read().split()  
		
            test_perplexity = float(a[a.index('logprob=')+1])
            #print("hypothesis %d with PPL %.3f" % (k,test_perplexity))

            ppl.append(test_perplexity)

        vals = np.array(ppl) + np.array(accoustic_score)	
        sort_index = np.argsort(vals)[::-1]
            
        sentences2 = []
        for k in sort_index:
            sen = sentences[k].split()
            sen[1] = str(ppl[k])
            sen = ' '.join(sen)
            sentences2.append(sen.encode('utf-8'))
            
        with open(output_file,'w') as h:
            for sentence in sentences2: h.write(sentence +'\n')

os.system('python compute_WER.py  --n_best ' +  output_dir + ' --name ' + name + '_WER')
    
print('done')

