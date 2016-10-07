import sys

with open('dagman_output.txt', 'a') as fid:
	fid.write(sys.argv[1] + " ")
