from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

#example documents
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
model = Doc2Vec(documents, dm=1, vector_size=10, window=2, min_count=1, workers=4)

model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

vector = model.infer_vector(['I', 'use', 'DBOW', 'for', 'word', 'sentence', 
                            'embeddings'])
print(vector)

