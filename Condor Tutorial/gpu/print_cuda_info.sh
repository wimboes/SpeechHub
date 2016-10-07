#! /bin/bash --norc

echo "I am $(hostname) and this is my CUDA info:"
echo
condor_status -long $(hostname) | grep CUDA
