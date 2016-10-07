import sys

with open('jobs_output.txt', 'a') as fid:
	fid.write('hello from process %s\n' % sys.argv[1])
