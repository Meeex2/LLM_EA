import torch
from tqdm import tqdm
import json


def load_embeddings(pkl_file, cuda=True):
    print(f"Loading embeddings from {pkl_file}...")
    if cuda:
        return torch.load("data/embeddings/" + pkl_file).cuda()
    else:
        return torch.load("data/embeddings/" + pkl_file)


def load_metadata(json_file):
    print(f"Loading metadata from {json_file}...")
    with open("data/embeddings/" + json_file, "r") as f:
        return json.load(f)


def load_mapping(mapping_file):
    mapping = {}
    with open(mapping_file, "r") as f:
        for line in f:
            uri1, uri2 = line.strip().split("\t")
            key1 = uri1.rsplit("/", 1)[-1].replace("_", " ")
            key2 = uri2.rsplit("/", 1)[-1].replace("_", " ")

            # Create bidirectional mapping
            mapping[key1] = key2
            mapping[key2] = key1
    return mapping


def get_max_or_zeros(tensor):
    if tensor.size(0) == 0:  # Check if the tensor is empty
        # Return a tensor of zeros with the same number of columns and device as the input tensor
        return torch.zeros(tensor.size(1), dtype=tensor.dtype, device=tensor.device)
    else:
        return torch.max(tensor, dim=0)[
            0
        ]  # Return the maximum along the specified dimension


def get_mean_or_zeros(tensor):
    if tensor.size(0) == 0:  # Check if the tensor is empty
        # Return a tensor of zeros with the same number of columns and device as the input tensor
        return torch.zeros(tensor.size(1), dtype=tensor.dtype, device=tensor.device)
    else:
        return torch.mean(
            tensor, dim=0
        )  # Return the maximum along the specified dimension


def aggregate_all_entities(metadata, entities, attributes, values, relationships):
    print("Aggregating embeddings for all entities...")
    all_embeddings = []

    for entity_name in tqdm(metadata.keys()):
        entity_idx = metadata[entity_name]["index"]
        entity_embedding = entities[entity_idx]

        # Aggregating attribute embeddings based on index ranges
        a, b = metadata[entity_name]["attributes_indices"]
        attribute_embeddings = get_mean_or_zeros(attributes[a:b])

        # Aggregating value embeddings based on index ranges
        a, b = metadata[entity_name]["values_indices"]
        value_embeddings = get_mean_or_zeros(values[a:b])

        # Aggregating relationship embeddings based on index ranges
        a, b = metadata[entity_name]["relationships_indices"]
        relationship_embeddings = get_mean_or_zeros(relationships[a:b])

        # Get neighbors
        neighbors = metadata[entity_name]["neighbors"]

        # Concatenate neighbors, if empty, use zeros
        if len(neighbors) == 0:
            neighbors_embeddings = torch.zeros(entity_embedding.size(0)).to("cuda")
        else:
            neighbors_embeddings = []
            for neighbor in neighbors:
                neighbor_idx = metadata[neighbor]["index"]
                neighbor_embedding = entities[neighbor_idx]
                neighbors_embeddings.append(neighbor_embedding)

            # Convert to tensor
            neighbors_embeddings = torch.stack(neighbors_embeddings)

            # Average neighbors
            neighbors_embeddings = torch.mean(neighbors_embeddings, dim=0)

        # Concatenate embeddings into one vector
        aggregated_embedding = torch.cat(
            (
                entity_embedding,
                attribute_embeddings,
                value_embeddings,
                relationship_embeddings,
                neighbors_embeddings,
            )
        )
        all_embeddings.append(aggregated_embedding)

    return torch.stack(all_embeddings)


def align_entities_with_cdist(
    entities_en_tensor,
    entities_fr_tensor,
    meta_en,
    meta_fr,
    mapping,
    top_k_values,
):
    print("Calculating distances between all entities...")
    num_en = len(meta_en)

    # Compute pairwise Euclidean distance between English and French embeddings
    distances = torch.cdist(entities_en_tensor, entities_fr_tensor, p=2)

    # Prepare to store results
    hits_at_k = {k: 0 for k in top_k_values}

    print(f"Aligning {num_en} English entities with French counterparts...")
    for i, entity_en in tqdm(enumerate(meta_en.keys()), total=num_en):
        # Find the top K closest French entities
        _, top_k_indices = torch.topk(
            distances[i],
            k=max(top_k_values),
            largest=False,
        )

        # Retrieve the correct match for this entity
        correct_match = mapping.get(entity_en)

        # Check for the correct match in top K results
        if correct_match:
            for k in top_k_values:
                top_k_matches = [
                    list(meta_fr.keys())[idx.item()] for idx in top_k_indices[:k]
                ]
                if correct_match in top_k_matches:
                    hits_at_k[k] += 1

    # Calculate Hit@K metrics
    hits_at_k = {k: hits / num_en for k, hits in hits_at_k.items()}

    print("Alignment process completed.")
    return hits_at_k


def main():
    # Load English and French embeddings
    entities_en = load_embeddings("entities_embeddings_dbp15k_en.pkl")
    entities_fr = load_embeddings("entities_embeddings_dbp15k_fr.pkl")
    attributes_en = load_embeddings("attributes_embeddings_dbp15k_en.pkl")
    attributes_fr = load_embeddings("attributes_embeddings_dbp15k_fr.pkl")
    relationships_en = load_embeddings("relationships_embeddings_dbp15k_en.pkl")
    relationships_fr = load_embeddings("relationships_embeddings_dbp15k_fr.pkl")
    values_en = load_embeddings("values_embeddings_dbp15k_en.pkl")
    values_fr = load_embeddings("values_embeddings_dbp15k_fr.pkl")

    # Load metadata for both languages
    meta_data_en = load_metadata("meta_data_dbp15k_en.json")
    meta_data_fr = load_metadata("meta_data_dbp15k_fr.json")

    # Aggregate embeddings for both English and French entities
    entities_en_tensor = aggregate_all_entities(
        meta_data_en, entities_en, attributes_en, values_en, relationships_en
    )
    del entities_en
    del attributes_en
    del values_en
    del relationships_en
    torch.cuda.empty_cache()

    entities_fr_tensor = aggregate_all_entities(
        meta_data_fr, entities_fr, attributes_fr, values_fr, relationships_fr
    )
    del entities_fr
    del attributes_fr
    del values_fr
    del relationships_fr
    torch.cuda.empty_cache()

    mapping_file = "data/ent_ILLs"
    mapping = load_mapping(mapping_file)

    top_k_values = [1, 5, 10, 20]

    hits_at_k_metrics = align_entities_with_cdist(
        entities_en_tensor,
        entities_fr_tensor,
        meta_data_en,
        meta_data_fr,
        mapping,
        top_k_values,
    )

    # Print the Hit@K metrics
    print("\nHit@K Metrics:")
    for k in top_k_values:
        print(f"Hit@{k}: {hits_at_k_metrics[k]:.4f}")


main()
