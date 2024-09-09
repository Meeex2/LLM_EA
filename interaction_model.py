import torch
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mapping = {}
with open("data/ent_ILLs", "r") as f:
    for line in f:
        uri1, uri2 = line.strip().split("\t")
        key1 = uri1.rsplit("/", 1)[-1].replace("_", " ")
        key2 = uri2.rsplit("/", 1)[-1].replace("_", " ")

        # Create bidirectional mapping
        mapping[key1] = key2
        mapping[key2] = key1

with open("data/embeddings/meta_data_dbp15k_en.json", "rb") as f:
    meta_data_en = json.load(f)

with open("data/embeddings/meta_data_dbp15k_fr.json", "rb") as f:
    meta_data_fr = json.load(f)

entities_en = torch.load(
    "data/embeddings/entities_embeddings_dbp15k_en.pkl", map_location=device
)
entities_fr = torch.load(
    "data/embeddings/entities_embeddings_dbp15k_fr.pkl", map_location=device
)

entities_similarity = torch.cdist(entities_en, entities_fr, p=2)

# Pick top k entities
k = 10
indices = entities_similarity.topk(k=k, dim=1, largest=False).indices.tolist()

# print entities
for j in range(10):
    print("===================================================================")
    entity = list(meta_data_en.values())[j]["name"]
    target = mapping[entity]
    print("entity:", entity)
    print("target:", target)
    print("target index:", list(meta_data_fr.keys()).index(target))
    print("===================================================================")
    for i in indices[j]:
        print("index: ", i, "  name: ", list(meta_data_fr.values())[i]["name"])


# Measure Hit(n)
def hit(n):
    hit = 0
    J = entities_en.shape[0]
    for j in range(J):
        entity = list(meta_data_en.values())[j]["name"]
        target = mapping[entity]
        try:
            target_index = list(meta_data_fr.keys()).index(target)
        except ValueError:
            continue
        if target_index in indices[j][:n]:
            hit += 1

    print(f"hit@{n}: ", hit / J)


print("Measuring hit@n...")
for n in [1, 3, 5, 10]:
    hit(n)
