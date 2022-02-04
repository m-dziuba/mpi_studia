#!/bin/bash

mpicxx main.cpp

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "Rozpoczynam kalukalcję dla 1 procesu"

mpiexec -n 1 a.out
echo "Zakończono"
echo ""
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "Rozpoczynam kalukalcję dla 2 procesów"

mpiexec -n 2 a.out
echo "Zakończono"
echo ""
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "Rozpoczynam kalukalcję dla 4 procesów"

mpiexec -n 4 a.out
echo "Zakończono"
echo ""
