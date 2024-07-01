#!/bin/bash
@echo off
conda activate fl
#com entropia
#python main.py -nc 100 -nb 10 -c cka -cp 100 -jr 0.2 -ws 1 -ncl 18 -ent 1 -gr 30 -t 5
#sem entropia mas com pesos
#python main.py -nc 100 -nb 10 -c cka -cp 100 -jr 0.2 -ws 1 -ncl 18 -wc 1 -gr 30 -t 5
#Sem pesos e sem entropia
#python main.py -nc 100 -nb 10 -cp 100 -jr 0.2 -ws 0 -gr 50 -t 5

#python main.py -nc 60 -nb 10 -gr 30 -c cka -cp 100  -t 5 -jr 0.3 -data "mnist" -ncl 10 -tsa A
#python main.py -nc 60 -nb 10 -gr 30 -c cka -cp 100  -t 5 -jr 0.3 -data "mnist" -ncl 10 -tsa D
#python main.py -nc 60 -nb 10 -gr 30  -t 5 -jr 0.3 -data "mnist"

############################################################################################################


#python main.py -nc 99 -nnc 1 -nb 10 -c cka -ncl 10 -gr 100 -cp 50 -jr 0.2 -tsa A -e 1 -wc 0 -ft 0.01 -t 10
#python main.py -nc 99 -nnc 1 -nb 10 -c cka -ncl 10 -gr 100 -cp 50 -jr 0.2 -tsa A -e 1 -wc 0 -ft 0.05 -t 10 -data Cifar10

#python main.py -nc 99 -nnc 1 -nb 10 -c cka -ncl 10 -gr 100 -cp 50 -jr 0.2 -tsa A -e 0 -wc 1 -ft 0.01 -t 10
#python main.py -nc 99 -nnc 1 -nb 10 -c cka -ncl 10 -gr 100 -cp 50 -jr 0.2 -tsa A -e 0 -wc 1 -ft 0.05 -t 10 -data Cifar10

#python main.py -nc 99 -nnc 1 -nb 10 -c cka -ncl 10 -gr 100 -cp 50 -jr 0.2 -tsa A -e 0 -wc 0 -ft 0.01 -t 10
#python main.py -nc 99 -nnc 1 -nb 10 -c cka -ncl 10 -gr 100 -cp 50 -jr 0.2 -tsa A -e 0 -wc 0 -ft 0.05 -t 10 -data Cifar10

############################################################################################################
"""

python main.py -nc 99 -nnc 1 -nb 10 -c cka -ncl 10 -gr 100 -cp 50 -jr 0.2 -tsa A -e 1 -wc 0 -ft 0.01 -t 10 -algo MOON
python main.py -nc 99 -nnc 1 -nb 10 -c cka -ncl 10 -gr 100 -cp 50 -jr 0.2 -tsa A -e 1 -wc 0 -ft 0.05 -t 10 -algo MOON

python main.py -nc 99 -nnc 1 -nb 10 -c cka -ncl 10 -gr 100 -cp 50 -jr 0.2 -tsa A -e 0 -wc 1 -ft 0.01 -t 10 -algo MOON
python main.py -nc 99 -nnc 1 -nb 10 -c cka -ncl 10 -gr 100 -cp 50 -jr 0.2 -tsa A -e 0 -wc 1 -ft 0.05 -t 10 -algo MOON

python main.py -nc 99 -nnc 1 -nb 10 -c cka -ncl 10 -gr 100 -cp 50 -jr 0.2 -tsa A -e 0 -wc 0 -ft 0.01 -t 10 -algo MOON
python main.py -nc 99 -nnc 1 -nb 10 -c cka -ncl 10 -gr 100 -cp 50 -jr 0.2 -tsa A -e 0 -wc 0 -ft 0.05 -t 10 -algo MOON

############################################################################################################

python main.py -nc 99 -nnc 1 -nb 10 -c cka -ncl 10 -gr 100 -cp 50 -jr 0.2 -tsa A -e 1 -wc 0 -ft 0.01 -t 10 -algo FedALA
python main.py -nc 99 -nnc 1 -nb 10 -c cka -ncl 10 -gr 100 -cp 50 -jr 0.2 -tsa A -e 1 -wc 0 -ft 0.05 -t 10 -algo FedALA

python main.py -nc 99 -nnc 1 -nb 10 -c cka -ncl 10 -gr 100 -cp 50 -jr 0.2 -tsa A -e 0 -wc 1 -ft 0.01 -t 10 -algo FedALA
python main.py -nc 99 -nnc 1 -nb 10 -c cka -ncl 10 -gr 100 -cp 50 -jr 0.2 -tsa A -e 0 -wc 1 -ft 0.05 -t 10 -algo FedALA

python main.py -nc 99 -nnc 1 -nb 10 -c cka -ncl 10 -gr 100 -cp 50 -jr 0.2 -tsa A -e 0 -wc 0 -ft 0.01 -t 10 -algo FedALA
python main.py -nc 99 -nnc 1 -nb 10 -c cka -ncl 10 -gr 100 -cp 50 -jr 0.2 -tsa A -e 0 -wc 0 -ft 0.05 -t 10 -algo FedALA
"""


#############################################################################################################
'''
python main.py -data fmnist -nc 50 -gr 100 -t 10 -tsa E -jr 0.2 -c CKA -ncl 3 -cf 1 -ncf 5 -rcf 1
python main.py -data fmnist -nc 50 -gr 100 -t 10 -tsa E -jr 0.2 -cf 1 -ncf 5
python main.py -data fmnist -nc 50 -gr 100 -t 10 -tsa A -jr 0.2 -cf 1 -ncf 5'''

#python main.py -data fmnist -nc 50 -gr 100 -t 10 -tsa A -jr 0.2 -cf 0
#python main.py -data Cifar100 -nc 50 -gr 100 -t 10 -tsa A -jr 0.2 -cf 0
#python main.py -data Cifar10 -nc 50 -gr 100 -t 10 -tsa A -jr 0.2 -cf 0

#python main.py -data Cifar100 -nc 50 -gr 100 -t 10 -tsa E -jr 0.2 -c CKA -ncl 3 -cf 1 -ncf 5 -rcf 1 -nb 100
#python main.py -data Cifar100 -nc 50 -gr 100 -t 10 -tsa E -jr 0.2 -cf 1 -ncf 5 -nb 100
#python main.py -data Cifar100 -nc 50 -gr 100 -t 10 -tsa A -jr 0.2 -cf 1 -ncf 5 -nb 100

#python main.py -data Cifar10 -nc 50 -gr 100 -t 10 -tsa E -jr 0.2 -c CKA -ncl 3 -cf 1 -ncf 5 -rcf 1
#python main.py -data Cifar10 -nc 50 -gr 100 -t 10 -tsa E -jr 0.2 -cf 1 -ncf 5
#python main.py -data Cifar10 -nc 50 -gr 100 -t 10 -tsa A -jr 0.2 -cf 1 -ncf 5


#############################################################################################################
"""
python main.py -nc 100 -gr 100 -t 10 -tsa E -jr 0.2 -c CKA -cf 1 -ncf 10 -rcf 1 -pcf 0.5 -ncl 2
python main.py -nc 100 -gr 100 -t 10 -tsa E -jr 0.2 -c CKA -cf 1 -ncf 10 -rcf 1 -pcf 0.5 -ncl 3
python main.py -nc 100 -gr 100 -t 10 -tsa E -jr 0.2 -c CKA -cf 1 -ncf 10 -rcf 1 -pcf 0.5 -ncl 5
python main.py -nc 100 -gr 100 -t 10 -tsa E -jr 0.2 -c CKA -cf 1 -ncf 10 -rcf 1 -pcf 0.5 -ncl 10
"""
"""python main.py -nc 50 -gr 100 -t 10 -tsa E -jr 0.2 -c CKA -cf 1 -ncf 5 -rcf 1 -pcf 0.25 -ncl 5
python main.py -nc 50 -gr 100 -t 10 -tsa E -jr 0.2 -c CKA -cf 1 -ncf 5 -rcf 1 -pcf 0.50 -ncl 5
python main.py -nc 50 -gr 100 -t 10 -tsa E -jr 0.2 -c CKA -cf 1 -ncf 5 -rcf 1 -pcf 0.75 -ncl 5
python main.py -nc 50 -gr 100 -t 10 -tsa E -jr 0.2 -c CKA -cf 1 -ncf 5 -rcf 1 -pcf 1.00 -ncl 5

python main.py -nc 50 -gr 100 -tsa E -jr 0.2 -c CKA -cf 1 -ncf 5 -rcf 1 -pcf 0.25 -ncl 5
python main.py -nc 50 -gr 100 -tsa E -jr 0.2 -c CKA -cf 1 -ncf 5 -rcf 1 -pcf 1 -ncl 5
python main.py -nc 50 -gr 100 -tsa E -jr 0.2 -c CKA -cf 1 -ncf 5 -rcf 1 -pcf 0.75 -ncl 5
python main.py -nc 50 -gr 100 -tsa E -jr 0.2 -c CKA -cf 1 -ncf 5 -rcf 1 -pcf 0.50 -ncl 5

python main.py -nc 50 -gr 100 -t 10 -tsa E -jr 0.2 -c CKA -cf 1 -ncf 5 -rcf 1 -pcf 1 -ncl 2
python main.py -nc 50 -gr 100 -t 10 -tsa E -jr 0.2 -c CKA -cf 1 -ncf 5 -rcf 1 -pcf 1 -ncl 3
python main.py -nc 50 -gr 100 -t 10 -tsa E -jr 0.2 -c CKA -cf 1 -ncf 5 -rcf 1 -pcf 1 -ncl 5
python main.py -nc 50 -gr 100 -t 10 -tsa E -jr 0.2 -c CKA -cf 1 -ncf 5 -rcf 1 -pcf 1 -ncl 10

python main.py -nc 50 -gr 40 -t 10 -tsa E -jr 0.2 -c CKA -cf 1 -ncf 5 -rcf 1 -pcf 0.25 -ncl 3
python main.py -nc 50 -gr 40 -t 10 -tsa E -jr 0.2 -c CKA -cf 1 -ncf 5 -rcf 1 -pcf 0.50 -ncl 3
python main.py -nc 50 -gr 40 -t 10 -tsa E -jr 0.2 -c CKA -cf 1 -ncf 5 -rcf 1 -pcf 0.75 -ncl 3
python main.py -nc 50 -gr 40 -t 10 -tsa E -jr 0.2 -c CKA -cf 1 -ncf 5 -rcf 1 -pcf 1.00 -ncl 3
"""
"""
python main.py -nc 50 -gr 100 -t 10 -tsa E -jr 0.2 -c CKA -cf 1 -ncf 10 -rcf 1 -pcf 1 -ncl 2 -data fmnist
python main.py -nc 50 -gr 100 -t 10 -tsa E -jr 0.2 -c CKA -cf 1 -ncf 10 -rcf 1 -pcf 1 -ncl 3 -data fmnist
python main.py -nc 50 -gr 100 -t 10 -tsa E -jr 0.2 -c CKA -cf 1 -ncf 10 -rcf 1 -pcf 1 -ncl 5 -data fmnist

python main.py -nc 50 -gr 100 -t 10 -tsa E -jr 0.2 -c CKA -cf 1 -ncf 10 -rcf 1 -pcf 1 -ncl 2 -data Cifar10
python main.py -nc 50 -gr 100 -t 10 -tsa E -jr 0.2 -c CKA -cf 1 -ncf 10 -rcf 1 -pcf 1 -ncl 3 -data Cifar10
python main.py -nc 50 -gr 100 -t 10 -tsa E -jr 0.2 -c CKA -cf 1 -ncf 10 -rcf 1 -pcf 1 -ncl 5 -data Cifar10

python main.py -nc 50 -gr 100 -t 10 -tsa E -jr 0.2 -c CKA -cf 1 -ncf 10 -rcf 1 -pcf 1 -ncl 2 -data mnist
python main.py -nc 50 -gr 100 -t 10 -tsa E -jr 0.2 -c CKA -cf 1 -ncf 10 -rcf 1 -pcf 1 -ncl 3 -data mnist
python main.py -nc 50 -gr 100 -t 10 -tsa E -jr 0.2 -c CKA -cf 1 -ncf 10 -rcf 1 -pcf 1 -ncl 5 -data mnist
"""
"""
cd ..
cd dataset
python generate_mnist.py -nc 20
python generate_fmnist.py -nc 20
python generate_cifar10.py -nc 20
cd ..
cd system

python main.py -data fmnist -nc 20 -gr 100 -t 10 -tsa E -jr 0.2 -c CKA -ncl 2 -cf 1 -ncf 2 -rcf 1 #rica + cka
python main.py -data fmnist -nc 20 -gr 100 -t 10 -tsa E -jr 0.2 -cf 1 -ncf 2 #rica
python main.py -data fmnist -nc 20 -gr 100 -t 10 -tsa A -jr 0.2 -cf 1 -ncf 2 #default

python main.py -data mnist -nc 20 -gr 100 -t 10 -tsa E -jr 0.2 -c CKA -ncl 2 -cf 1 -ncf 2 -rcf 1 #rica + cka
python main.py -data mnist -nc 20 -gr 100 -t 10 -tsa E -jr 0.2 -cf 1 -ncf 2 #rica
python main.py -data mnist -nc 20 -gr 100 -t 10 -tsa A -jr 0.2 -cf 1 -ncf 2 #default

python main.py -data Cifar10 -nc 20 -gr 100 -t 10 -tsa E -jr 0.2 -c CKA -ncl 2 -cf 1 -ncf 2 -rcf 1 #rica + cka
python main.py -data Cifar10 -nc 20 -gr 100 -t 10 -tsa E -jr 0.2 -cf 1 -ncf 2 #rica
python main.py -data Cifar10 -nc 20 -gr 100 -t 10 -tsa A -jr 0.2 -cf 1 -ncf 2 #default


cd ..
cd dataset
python generate_mnist.py -nc 30
python generate_fmnist.py -nc 30
python generate_cifar10.py -nc 30
cd ..
cd system

python main.py -data fmnist -nc 30 -gr 100 -t 10 -tsa E -jr 0.2 -c CKA -ncl 2 -cf 1 -ncf 3 -rcf 1 #rica + cka
python main.py -data fmnist -nc 30 -gr 100 -t 10 -tsa E -jr 0.2 -cf 1 -ncf 3 #rica
python main.py -data fmnist -nc 30 -gr 100 -t 10 -tsa A -jr 0.2 -cf 1 -ncf 3 #default

python main.py -data mnist -nc 30 -gr 100 -t 10 -tsa E -jr 0.2 -c CKA -ncl 2 -cf 1 -ncf 3 -rcf 1 #rica + cka
python main.py -data mnist -nc 30 -gr 100 -t 10 -tsa E -jr 0.2 -cf 1 -ncf 3 #rica
python main.py -data mnist -nc 30 -gr 100 -t 10 -tsa A -jr 0.2 -cf 1 -ncf 3 #default

python main.py -data Cifar10 -nc 30 -gr 100 -t 10 -tsa E -jr 0.2 -c CKA -ncl 2 -cf 1 -ncf 3 -rcf 1 #rica + cka
python main.py -data Cifar10 -nc 30 -gr 100 -t 10 -tsa E -jr 0.2 -cf 1 -ncf 3 #rica
python main.py -data Cifar10 -nc 30 -gr 100 -t 10 -tsa A -jr 0.2 -cf 1 -ncf 3 #default


cd ..
cd dataset
python generate_mnist.py -nc 40
python generate_fmnist.py -nc 40
python generate_cifar10.py -nc 40
cd ..
cd system

python main.py -data fmnist -nc 40 -gr 100 -t 10 -tsa E -jr 0.2 -c CKA -ncl 3 -cf 1 -ncf 4 -rcf 1 #rica + cka
python main.py -data fmnist -nc 40 -gr 100 -t 10 -tsa E -jr 0.2 -cf 1 -ncf 4 #rica
python main.py -data fmnist -nc 40 -gr 100 -t 10 -tsa A -jr 0.2 -cf 1 -ncf 4 #default

python main.py -data mnist -nc 40 -gr 100 -t 10 -tsa E -jr 0.2 -c CKA -ncl 3 -cf 1 -ncf 4 -rcf 1 #rica + cka
python main.py -data mnist -nc 40 -gr 100 -t 10 -tsa E -jr 0.2 -cf 1 -ncf 4 #rica
python main.py -data mnist -nc 40 -gr 100 -t 10 -tsa A -jr 0.2 -cf 1 -ncf 4 #default

python main.py -data Cifar10 -nc 40 -gr 100 -t 10 -tsa E -jr 0.2 -c CKA -ncl 3 -cf 1 -ncf 4 -rcf 1 #rica + cka
python main.py -data Cifar10 -nc 40 -gr 100 -t 10 -tsa E -jr 0.2 -cf 1 -ncf 4 #rica
python main.py -data Cifar10 -nc 40 -gr 100 -t 10 -tsa A -jr 0.2 -cf 1 -ncf 4 #default

"""

python main.py -data Cifar10 -nc 30 -gr 100 -t 10 -tsa A -jr 0.2 -cf 1 -ncf 3 #default
python main.py -data fmnist -nc 30 -gr 100 -t 10 -tsa E -jr 0.2 -c CKA -ncl 2 -cf 1 -ncf 3 -rcf 1 #rica + cka
python main.py -data mnist -nc 30 -gr 100 -t 7 -tsa E -jr 0.2 -c CKA -ncl 2 -cf 1 -ncf 3 -rcf 1 #rica + cka
python main.py -data Cifar10 -nc 30 -gr 100 -t 5 -tsa E -jr 0.2 -c CKA -ncl 2 -cf 1 -ncf 3 -rcf 1 #rica + cka
python main.py -data Cifar10 -nc 30 -gr 100 -t 5 -tsa E -jr 0.2 -cf 1 -ncf 3 #rica

