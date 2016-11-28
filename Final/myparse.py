#!/usr/bin/env python
# -*- coding: utf-8 -*-
import  numpy as np
import  csv

def readcsv(input_file):
    print "Reading "+input_file
    with open(input_file,'rb') as csvfile:
        data_iter = csv.reader(csvfile,delimiter=',')
        data = [data for data in data_iter]
    data_array = np.asarray(data)
    return data_array

def writecsv(output_file, dic_to_dump):
    print "Writing "+output_file
    with open(output_file,'wb') as csvfile:
        fieldnames=dic_to_dump[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dic_to_dump)

def yieldlines(thefile, whatlines):
    return (x for i, x in enumerate(thefile) if i in whatlines)

#month_total = np.array([625457, 627394, 629209])
month_total = np.array([625457, 627394, 629209, 630367, 631957, 632110, 829817, 843201, 865440, 892251, 906109, 912021, 916269, 920904, 925076, 928274, 931453])
start = 1
for i,num in enumerate(month_total):
    file_in  = open("data/train_ver2.csv",'r')
    # first_line should be in splits (headers!!)
    index = range(start, start + num)
    index.append(0)
    whatlines = set(index)
    file_out = open("data/month"+str(i+1)+"_"+str(num)+".csv",'w')
    for lines in yieldlines(file_in, whatlines):
        file_out.write(lines)
    file_in.close()
    file_out.close()
    start += num
#readcsv("object.csv")
#readcsv("log_train.csv")
#readcsv("sample_train_x.csv")

