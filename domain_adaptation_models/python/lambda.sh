#! /bin/bash --norc

compute-best-mix interpol0.txt interpol1.txt | grep -o "best lambda.*" | grep -o "(.*)" | grep -o "[^(].*[^)]" > lambda.txt
