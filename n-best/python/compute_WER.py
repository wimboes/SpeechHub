import argparse
import os
import shutil
import fnmatch

##### flags

python_path = os.path.abspath(os.getcwd())
general_path = os.path.split(os.path.split(python_path)[0])[0]
data_path = os.path.join(general_path,'input_n_best')

ap = argparse.ArgumentParser()
ap.add_argument('--ref', default = os.path.join(data_path, 'reference') , type=str)
ap.add_argument('--n_best', default=os.path.join(data_path, 'original_n-best'), type=str)
ap.add_argument('--out', default=os.path.join(os.path.split(python_path)[0], 'output'), type=str)
ap.add_argument('--name', default='original_n_best', type=str)

opts = ap.parse_args()
ref = opts.ref
n_best = opts.n_best
out_folder = os.path.join(opts.out, opts.name)

def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))

if (os.path.exists(out_folder)):
    remove(out_folder) 
    os.mkdir(out_folder)
else:
    os.mkdir(out_folder)

n_best_files = [file for file in os.listdir(n_best) if fnmatch.fnmatch(file, 'fv*')]
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

with open(os.path.join(out_folder, 'reference.txt'), 'w') as reference, open(os.path.join(out_folder, 'recognized.txt'), 'w') as recognized:
    for i in range(len(fv_files)):
	with open(os.path.join(ref,fv_files[i] + '.stm'), 'r') as f:
    	    reco = f.readline()
	reference.write(reco)
	sentence = ''
        for j in range(fv_files_amount[i]+1):
	    with open(os.path.join(n_best,fv_files[i] + '.0.' + str(j) + '.10000-best.txt'), 'r') as f:
    	        line = f.readline().split()[4:-1]
		sentence += ' '.join([word for word in line if word != '<sil>']) + ' '
	recognized.write(sentence + '\n')

os.system('/users/spraak/spchprog/SPRAAK/v1.2/bin/Linux_x86_64/spr_scoreres -ref ' + os.path.join(out_folder, 'reference.txt') + ' -tst ' +  os.path.join(out_folder, 'recognized.txt') + ' -PAR -nr ' + os.path.join(out_folder, 'WER.txt'))
		
os.system('/users/spraak/spchprog/SPRAAK/v1.2/bin/Linux_x86_64/spr_scoreres -ref ' + os.path.join(out_folder, 'reference.txt') + ' -tst ' +  os.path.join(out_folder, 'recognized.txt') + ' -PAR')
