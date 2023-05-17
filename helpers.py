import os
import pickle as pkl
import torch


def load_known_embeddings():
    """Возвращает список 'знакомых характеристик' голосов"""
    result = []
    emb_dir = "known_embeddings"
    for file_name in os.listdir(emb_dir):
        result.append(pkl.load(open(f"{emb_dir}/{file_name}", "rb")))
    return result


def verify_embeddings(emb1, emb2, threshold):
    """Сравнивает две характеристики,
       возвращает True, если схожесть больше порогового значения"""
    embs1 = emb1.squeeze()
    embs2 = emb2.squeeze()

    X = embs1 / torch.linalg.norm(embs1)
    Y = embs2 / torch.linalg.norm(embs2)

    similarity_score = torch.dot(X, Y) / ((torch.dot(X, X) * torch.dot(Y, Y)) ** 0.5)
    similarity_score = (similarity_score + 1) / 2

    if similarity_score >= threshold:
        return True
    else:
        return False


def clear_cache(cache_dir_path): #пока не используется
    """Очищает папку кеша от всех файлов"""
    for file_name in os.listdir(cache_dir_path):
        os.remove(f"{cache_dir_path}/{file_name}")


def save_known_embeddings(known_embeddings):
    """Сохраняет все известные характеристики из поданного списка в виде .pkl файлов"""
    for i in range(len(known_embeddings)):
        pkl.dump(known_embeddings[i], open(f"known_embeddings/saved_embedding{i}.pkl", "wb"))