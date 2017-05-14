import os

lst = [file for file in os.listdir(".") if file != 'transform.py']



lst = [name.split('.') for name in lst]



for i in range(len(lst)):
	lst[i].insert(2, int(lst[i][2].split(':')[0]))

lst.sort()

current_name = lst[0][0]
number = 0
for filename in lst:
	if filename[0] == current_name:
		f1 = '.'.join(filename[0:2]) +'.' + '.'.join(filename[3:6])
		f2 = '.'.join(filename[0:2]) +'.' + str(number) + '.' + '.'.join(filename[4:6])
		os.rename(f1,f2)
		number += 1
	else:
		current_name = filename[0]
		number = 0
		f1 = '.'.join(filename[0:2]) +'.' + '.'.join(filename[3:6])
		f2 = '.'.join(filename[0:2]) +'.' + str(number) + '.' + '.'.join(filename[4:6])
		os.rename(f1,f2)
		number += 1
