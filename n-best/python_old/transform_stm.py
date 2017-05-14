import os
import fnmatch

lst = [file for file in os.listdir('.') if fnmatch.fnmatch(file, '*.stm')]

for file in lst:
	with open(file, 'r') as f:
		sentences = [line.split() for line in f]
	print(sentences)
	sentence = ''
	for line in sentences:
		sentence += ' '.join(line[6:]) + ' '
	with open(file, 'w') as f:
		f.write(sentence + '\n')
