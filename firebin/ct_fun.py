#!/usr/bin/env pythonfx

namepy = ['pet.py', 'utils.py', 'viewer.py']
namec  = ['kernel.cpp']
namecu = ['kernel_cuda.cu']
nbline = 0

print '== Python functions ==\n'
ctpy = 0
for name in namepy:
    data    = open('../lib/' + name, 'r').readlines()
    nbline += len(data)
    for line in data:
        val = line.find('def ')
        if val == 0:
            print '%03i ' % ctpy, line.strip('\n')
            ctpy += 1

print '\n== C functions ==\n'            
ctc = 0            
for name in namec:
    data    = open('../lib/' + name, 'r').readlines()
    nbline += len(data)
    for line in data:
        val1 = line.find('void ')
        val2 = line.find('int ')
        val3 = line.find('float ')
        if val1==0 or val2==0 or val3==0:
            print '%03i' % ctc, line.strip('\n')
            ctc += 1

print '\n== CUDA functions ==\n'            
ctcu = 0
for name in namecu:
    data    = open('../lib/' + name, 'r').readlines()
    nbline += len(data)
    for line in data:
        val1 = line.find('void ')
        val2 = line.find('__global__ void ')
        if val1==0 or val2==0:
            print '%03i' % ctcu, line.strip('\n')
            ctcu += 1

print '\n::::: Tot :::::\n'
print '%03i Python functions' % ctpy
print '%03i C functions' % ctc
print '%03i CUDA functions' % ctcu
print '%i functions' % (ctpy+ctc+ctcu)
print '%i code lines' % nbline

