#! /bin/bash --norc

compute-best-mix ../output/topic_0/interpol0.txt ../output/topic_0/interpol1.txt | grep -o "best lambda.*" | grep -o "(.*)" | grep -o "[^(].*[^)]" > lambda.txt
