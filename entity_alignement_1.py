import torch
import json


# Load embeddings (tensors)
def load_embeddings(pkl_file):
    print(f"Loading embeddings from {pkl_file}...")
    return torch.load("data/embeddings/" + pkl_file).cuda()  # Load directly to GPU


# Load metadata
def load_metadata(json_file):
    print(f"Loading metadata from {json_file}...")
    with open("data/embeddings/" + json_file, "r") as f:
        return json.load(f)


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


# Aggregate embeddings for an entity, considering range indices
def aggregate_embeddings(
    entity_name, metadata, entities, attributes, values, relationships
):
    entity_idx = metadata[entity_name]["index"]
    entity_embedding = entities[entity_idx]

    # Aggregating attribute embeddings based on index ranges
    a, b = metadata[entity_name]["attributes_indices"]
    attribute_embeddings = torch.mean(attributes[a:b], dim=0)

    # Aggregating value embeddings based on index ranges
    a, b = metadata[entity_name]["values_indices"]
    value_embeddings = torch.mean(values[a:b], dim=0)

    # Aggregating relationship embeddings based on index ranges
    a, b = metadata[entity_name]["relationships_indices"]
    relationship_embeddings = torch.mean(relationships[a:b], dim=0)

    # Combine all embeddings
    aggregated_embedding = torch.cat(
        (
            entity_embedding,
            attribute_embeddings,
            value_embeddings,
            relationship_embeddings,
        )
    )
    return aggregated_embedding


# Compute cosine similarity using PyTorch
def compute_similarity(embedding1, embedding2):
    # Normalizing the embeddings
    embedding1 = embedding1 / embedding1.norm(dim=0)
    embedding2 = embedding2 / embedding2.norm(dim=0)
    # Dot product gives cosine similarity after normalization
    return torch.dot(embedding1, embedding2).item()


# Align entities by finding the most similar pairs
def align_entities(
    meta_en,
    meta_fr,
    entities_en,
    entities_fr,
    attributes_en,
    attributes_fr,
    values_en,
    values_fr,
    relationships_en,
    relationships_fr,
):
    alignments = []
    total_entities = len(meta_en)

    print(f"Aligning {total_entities} English entities with French counterparts...")

    for idx, entity_en in enumerate(meta_en.keys(), 1):
        print(f"\nProcessing entity {idx}/{total_entities}: {entity_en}")
        best_match = None
        best_similarity = -1

        en_embedding = aggregate_embeddings(
            entity_en, meta_en, entities_en, attributes_en, values_en, relationships_en
        )

        for entity_fr in meta_fr.keys():
            fr_embedding = aggregate_embeddings(
                entity_fr,
                meta_fr,
                entities_fr,
                attributes_fr,
                values_fr,
                relationships_fr,
            )
            similarity = compute_similarity(en_embedding, fr_embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entity_fr

        print(
            f"Best match for {entity_en} is {best_match} with similarity {best_similarity:.4f}"
        )
        alignments.append((entity_en, best_match, best_similarity))

    print("\nAlignment process completed.")
    return alignments


# Run the alignment process
aligned_entities = align_entities(
    meta_data_en,
    meta_data_fr,
    entities_en,
    entities_fr,
    attributes_en,
    attributes_fr,
    values_en,
    values_fr,
    relationships_en,
    relationships_fr,
)

# Print the results
print("\nTop 10 aligned entities:")
for en_entity, fr_entity, sim_score in aligned_entities[:10]:
    print(f"English: {en_entity}, French: {fr_entity}, Similarity: {sim_score:.4f}")
