from tqdm import tqdm
import openai
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


def embed_texts(texts, model="text-embedding-ada-002"):
    embeddings = []
    n_texts = len(texts)
    n_batches = n_texts // 20 + int(n_texts % 20 != 0)
    for i in tqdm.tqdm(range(n_batches)):
        for j in range(50):
            try:
                texts_subset = texts[20 * i : min(20 * (i + 1), n_texts)]
                embedding = openai.Embedding.create(
                    model="text-embedding-ada-002", input=texts_subset
                )
                break
            except:
                print("Skipping server error number " + str(j))
                time.sleep(2)
        embeddings += [e["embedding"] for e in embedding["data"]]
    return embeddings


def joint_embedding(
    inputs, responses, model="text-embedding-ada-002", combine_statements=False
):
    if not combine_statements:
        inputs_embeddings = embed_texts(inputs)
        responses_embeddings = embed_texts(responses)
        joint_embeddings = [
            i + r for i, r in zip(inputs_embeddings, responses_embeddings)
        ]
    else:
        joint_embeddings = embed_texts(
            [input + response for input, response in zip(inputs, responses)]
        )
    return joint_embeddings


def cluster_embeddings(embeddings, n_clusters=100):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings)
    return kmeans


def reduce_dimensions(embeddings, perplexity=30, n_iter=1000):
    tsne = TSNE(perplexity=perplexity, n_iter=n_iter)
    return tsne.fit_transform(embeddings)
