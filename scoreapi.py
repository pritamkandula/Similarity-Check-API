from fastapi import FastAPI
import uvicorn
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

app = FastAPI()

@app.get('/')
def main():
    return {'text':'Similarity Score of two sentences'}

@app.get('/predict-score')
def score(sentence1:str, sentence2:str):
    model = SentenceTransformer("bert-base-nli-mean-tokens")
    sentences = [sentence1, sentence2]
    embeddings = model.encode(sentences)
    embeddings.shape 
    output = cosine_similarity([embeddings[0]], embeddings[1:])
    return {'The similarity score for the given two sentences is {}'.format(output)}


if __name__ == '__main__':
    uvicorn.run(app)







