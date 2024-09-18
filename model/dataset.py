import json
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


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


def load_embeddings(pkl_file, cuda=True):
    print(f"Loading embeddings from {pkl_file}...")
    if cuda:
        return torch.load("data/embeddings/" + pkl_file, weights_only=True).cuda()
    else:
        return torch.load("data/embeddings/" + pkl_file, weights_only=True).cpu()


def load_metadata(json_file):
    print(f"Loading metadata from {json_file}...")
    with open("data/embeddings/" + json_file, "r") as f:
        return json.load(f)


def farthest_cosine_similarity(e, entities, names):
    """
    Finds the name of the entity in a batch of tensors with the farthest cosine similarity to vector e.

    Args:
        e (torch.Tensor): The reference vector.
        entities (torch.Tensor): A batch of entity vectors.
        names (list): List of names corresponding to the entity vectors.

    Returns:
        str: The name of the entity with the farthest cosine similarity.
    """
    # Normalize the reference vector and entities
    e_norm = F.normalize(e.unsqueeze(0), p=2, dim=1)
    entities_norm = F.normalize(entities, p=2, dim=1)

    # Compute cosine similarity
    cosine_similarities = torch.mm(e_norm, entities_norm.t())

    # Find the index of the minimum similarity (farthest)
    _, farthest_index = cosine_similarities.min(dim=1)

    # Return the name of the farthest entity
    return names[farthest_index.item()]


class GraphPairDataset(Dataset):
    def __init__(self, num_neg_samples):
        self.num_neg_samples = num_neg_samples

        self.ground_truth_pairs = self.load_ground_truth("data/ent_ILLs")

        self.entities_en = load_embeddings(
            "entities_embeddings_dbp15k_en.pkl", cuda=False
        )
        self.entities_fr = load_embeddings(
            "entities_embeddings_dbp15k_fr.pkl", cuda=False
        )
        self.attributes_en = load_embeddings(
            "attributes_embeddings_dbp15k_en.pkl", cuda=False
        )
        self.attributes_fr = load_embeddings(
            "attributes_embeddings_dbp15k_fr.pkl", cuda=False
        )
        self.relationships_en = load_embeddings(
            "relationships_embeddings_dbp15k_en.pkl", cuda=False
        )
        self.relationships_fr = load_embeddings(
            "relationships_embeddings_dbp15k_fr.pkl", cuda=False
        )
        self.values_en = load_embeddings("values_embeddings_dbp15k_en.pkl", cuda=False)
        self.values_fr = load_embeddings("values_embeddings_dbp15k_fr.pkl", cuda=False)

        # Load metadata for both languages
        self.meta_data_en = load_metadata("meta_data_dbp15k_en.json")
        self.meta_data_fr = load_metadata("meta_data_dbp15k_fr.json")

        # Prepare list of all English entities for negative sampling
        self.all_en_entities = list(self.meta_data_en.keys())
        self.all_fr_entities = list(self.meta_data_fr.keys())

    def load_ground_truth(self, ground_truth_file):
        pairs = []
        with open(ground_truth_file, "r") as f:
            for line in f:
                uri1, uri2 = line.strip().split("\t")
                key1 = uri1.rsplit("/", 1)[-1].replace("_", " ")
                key2 = uri2.rsplit("/", 1)[-1].replace("_", " ")
                pairs.append((key1, key2))
        return pairs

    def __len__(self):
        return int(len(self.ground_truth_pairs))  # Adjust size based on ratio

    def __getitem__(self, idx):
        entity_fr_name, entity_en_name = self.ground_truth_pairs[idx]

        # Prepare node data for the sample
        def get_node_data(
            metadata, entities, attributes, values, relationships, node_name
        ):
            data = metadata[node_name]
            idx = data["index"]
            attr_indices = data["attributes_indices"]
            val_indices = data["values_indices"]
            rel_indices = data["relationships_indices"]

            node_embedding = entities[idx]
            # Check if slices are empty and use zeros if so
            attr_embeddings = (
                attributes[attr_indices[0] : attr_indices[1]]
                if attr_indices[0] < attr_indices[1]
                else torch.zeros((1, attributes.shape[1]), device=attributes.device)
            )
            val_embeddings = (
                values[val_indices[0] : val_indices[1]]
                if val_indices[0] < val_indices[1]
                else torch.zeros((1, values.shape[1]), device=values.device)
            )
            rel_embeddings = (
                relationships[rel_indices[0] : rel_indices[1]]
                if rel_indices[0] < rel_indices[1]
                else torch.zeros(
                    (1, relationships.shape[1]), device=relationships.device
                )
            )

            return (
                node_embedding,
                torch.mean(attr_embeddings, dim=0),
                torch.mean(val_embeddings, dim=0),
                torch.mean(rel_embeddings, dim=0),
            )

        node_en = get_node_data(
            self.meta_data_en,
            self.entities_en,
            self.attributes_en,
            self.values_en,
            self.relationships_en,
            entity_en_name,
        )
        node_fr = get_node_data(
            self.meta_data_fr,
            self.entities_fr,
            self.attributes_fr,
            self.values_fr,
            self.relationships_fr,
            entity_fr_name,
        )

        # Get farthest entity fr name
        # neg_entity_fr_name = farthest_cosine_similarity(
        #     node_fr[0], self.entities_fr, self.all_fr_entities
        # )
        neg_entities_fr_names_list = [
            random.choice(self.all_fr_entities) for _ in range(self.num_neg_samples)
        ]

        neg_node_fr_list = [
            get_node_data(
                self.meta_data_fr,
                self.entities_fr,
                self.attributes_fr,
                self.values_fr,
                self.relationships_fr,
                name,
            )
            for name in neg_entities_fr_names_list
        ]

        (
            entity_en,
            attributes_en,
            values_en,
            relationships_en,
        ) = node_en
        (
            entity_fr,
            attributes_fr,
            values_fr,
            relationships_fr,
        ) = node_fr

        # (
        #     neg_entity_fr,
        #     neg_attributes_fr,
        #     neg_values_fr,
        #     neg_relationships_fr,
        # ) = neg_node_fr

        sample = {
            # English
            "e1": torch.cat(
                [entity_en, attributes_en, values_en, relationships_en], dim=0
            ),
            # French
            "e2": torch.cat(
                [entity_fr, attributes_fr, values_fr, relationships_fr], dim=0
            ),
            # # Negative French
            # "neg_e2": torch.cat(
            #     [neg_entity_fr, neg_attributes_fr, neg_values_fr, neg_relationships_fr],
            #     dim=0,
            # ),
            # Names
            "e1_name": entity_en_name,
            "e2_name": entity_fr_name,
            # "neg_e2_name": neg_entity_fr_name,
        }

        for i, node in enumerate(neg_node_fr_list):
            (
                neg_entity_fr,
                neg_attributes_fr,
                neg_values_fr,
                neg_relationships_fr,
            ) = node
            sample[f"neg_e2_{i}"] = torch.cat(
                [
                    neg_entity_fr,
                    neg_attributes_fr,
                    neg_values_fr,
                    neg_relationships_fr,
                ],
                dim=0,
            )

        return sample


def get_data_loaders(batch_size, num_neg_samples):
    # Create dataset and dataloader
    dataset = GraphPairDataset(num_neg_samples)
    dataset_size = len(dataset)

    # Define split ratios
    train_ratio = 0.6
    val_ratio = 0.2

    # Calculate the sizes of each dataset
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create DataLoaders for each set
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader
