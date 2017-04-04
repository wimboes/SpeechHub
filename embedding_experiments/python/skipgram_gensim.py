import gensim
import os

input_path = os.path.join(os.path.split(os.path.split(os.path.abspath(os.getcwd()))[0])[0],'input')
filename = 'ds.train.txt'

class sentence_generator():
    def __init__(self, input_path, filename):
        self.file_path = os.path.join(input_path,filename)
    def __iter__(self):
        for line in open(self.file_path,'r'):
            yield line.split()

model = gensim.models.Word2Vec(sentences=sentence_generator(input_path,filename), size=128, window=5, min_count=0, sg=1)
model.save(os.path.join(os.path.split(os.path.abspath(os.getcwd()))[0],'output/embedding_skip/embedding_skip.emb'))
