import argparse
import re
import sys
import numpy
import os
import sys
import pickle
files = sys.argv[1:]
#ipickled = [pickle.load(open(f, "rb")) for f in files]
pickled=[]
for f in files:
	print(f)
	pickled.append(pickle.load(open(f, "rb")))

def mean(x):
	return sum(x) / len(x)


def compute_f1(overlap, model_out, std_out):
    prec = float(len(overlap)) / (len(model_out) + 1e-8)
    reca = float(len(overlap)) / (len(std_out) + 1e-8)
    if len(std_out) == 0:
        reca = 1.
        if len(model_out) == 0:
            prec = 1.
    f1 = 2 * prec * reca / (prec + reca + 1e-8)
    return f1

for i in range(0, len(files)):
	for j in range(i+1, len(files)):
		print("SELF F1 B/W " + files[i] + " and " + files[j]+ "\n")
		f1_list=[]
		for ind in range(0,len(pickled[i])):
			overlap = pickled[i][ind].intersection(pickled[j][ind])
			f1_list.append(compute_f1(overlap,pickled[i][ind] , pickled[j][ind]))
		print(mean(f1_list))
