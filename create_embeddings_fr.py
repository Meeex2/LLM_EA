import json
from sentence_transformers import SentenceTransformer
import torch

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
model_minilm = SentenceTransformer("all-MiniLM-L6-v2").to(device)

with open("data/records_dbp15k_fr_clean.json", "r", encoding="utf-8-sig") as f:
    fr_data = json.load(f)

with open("data/rel_dbp15k_fr.json", "r", encoding="utf-8-sig") as f:
    fr_rel_data = json.load(f)

# Create dictionaries for quick lookup
fr_rel_dict = {item["source_uri"]: item for item in fr_rel_data}

entities = []
attributes = []
values = []
relationships = []

entities_embeddings = []
attributes_embeddings = []
values_embeddings = []
relationships_embeddings = []

meta_data = {}
index = 0
print("Preparing data...")
for data, rel_data in zip(fr_data, fr_rel_data):
    meta_ = {}

    # Take entity name
    entity_name = data["n.uri"].split("/")[-1].replace("_", " ")
    meta_["name"] = entity_name
    meta_["index"] = index
    entities.append(data["n.uri"].split("/")[-1].replace("_", " "))

    # Take keys from properties
    entity_attributes = data["properties(n)"].keys()
    # Convert to string
    entity_attributes = [str(k) for k in entity_attributes]
    meta_["attributes_indices"] = (
        len(attributes),
        len(attributes) + len(entity_attributes),
    )
    attributes.extend(entity_attributes)

    # Take values from properties
    entity_values = data["properties(n)"].values()
    # Convert to string
    entity_values = [str(v) for v in entity_values]
    meta_["values_indices"] = (len(values), len(values) + len(entity_values))
    values.extend(entity_values)

    # Take outgoing and incoming relationships
    relationships_values = [
        f"{r['type']}: {r['neighbor_uri'].split('/')[-1].replace('_', ' ')}"
        for r in rel_data.get("outgoing_relationships", [])
        if r["neighbor_uri"]
    ] + [
        f"{r['type']}: {r['neighbor_uri'].split('/')[-1].replace('_', ' ')}"
        for r in rel_data.get("incoming_relationships", [])
        if r["neighbor_uri"]
    ]
    meta_["relationships_indices"] = (
        len(relationships),
        len(relationships) + len(relationships_values),
    )
    relationships.extend(relationships_values)

    # Add neighbor names
    meta_["neighbors"] = [
        r["neighbor_uri"].split("/")[-1].replace("_", " ")
        for r in rel_data.get("outgoing_relationships", [])
        if r["neighbor_uri"]
    ] + [
        r["neighbor_uri"].split("/")[-1].replace("_", " ")
        for r in rel_data.get("incoming_relationships", [])
        if r["neighbor_uri"]
    ]

    meta_data[entity_name] = meta_
    index += 1

# Compute embeddings
print("Computing embeddings...")
print("Entities...")
entities_embeddings = model_minilm.encode(
    entities, convert_to_tensor=True, device=device, show_progress_bar=True
)
print("saving...")
with open("data/embeddings/entities_embeddings_dbp15k_fr.pkl", "wb") as f:
    torch.save(entities_embeddings, f)

print("Attributes...")
attributes_embeddings = model_minilm.encode(
    attributes, convert_to_tensor=True, device=device, show_progress_bar=True
)
print("saving...")
with open("data/embeddings/attributes_embeddings_dbp15k_fr.pkl", "wb") as f:
    torch.save(attributes_embeddings, f)

print("Values...")
values_embeddings = model_minilm.encode(
    values, convert_to_tensor=True, device=device, show_progress_bar=True
)
print("saving...")
with open("data/embeddings/values_embeddings_dbp15k_fr.pkl", "wb") as f:
    torch.save(values_embeddings, f)

print("Relationships...")
relationships_embeddings = model_minilm.encode(
    relationships,
    convert_to_tensor=True,
    device=device,
    show_progress_bar=True,
)
print("saving...")
with open("data/embeddings/relationships_embeddings_dbp15k_fr.pkl", "wb") as f:
    torch.save(relationships_embeddings, f)

# Save metadata
json.dump(meta_data, open("data/embeddings/meta_data_dbp15k_fr.json", "w"), indent=2)
