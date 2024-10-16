import torch
from tqdm import tqdm

from dataset import get_data_loaders
from node_alignement_model import NodeAlignmentModel


def get_scores(model, test_loader):
    model.eval()
    device = next(model.parameters()).device

    all_scores = torch.zeros(
        len(test_loader.dataset), len(test_loader.dataset), device="cpu"
    )

    with torch.no_grad():
        for i, batch1 in enumerate(tqdm(test_loader, desc="Processing batches")):
            e1 = batch1["e1"].to(device)

            for j, batch2 in enumerate(test_loader):
                e2 = batch2["e2"].to(device)

                # Use expand instead of cartesian_prod for memory efficiency
                e1_expanded = e1.unsqueeze(1).expand(-1, e2.size(0), -1)
                e2_expanded = e2.unsqueeze(0).expand(e1.size(0), -1, -1)

                # Reshape for model input
                e1_flat = e1_expanded.reshape(-1, e1.size(-1))
                e2_flat = e2_expanded.reshape(-1, e2.size(-1))

                # Get model predictions
                scores = model(e1_flat, e2_flat)
                scores = scores.view(e1.size(0), e2.size(0)).cpu()

                # Add scores to all_scores
                all_scores[
                    i * e1.size(0) : (i + 1) * e1.size(0),
                    j * e2.size(0) : (j + 1) * e2.size(0),
                ] = scores

    # Ensure we have square matrix of scores
    assert all_scores.size(0) == all_scores.size(1), "Score matrix should be square"
    return all_scores


def calculate_hits_at_n(all_scores, N):
    num_english_entities = all_scores.shape[0]

    # Correct indices are assumed to be on the diagonal
    correct_indices = torch.arange(num_english_entities)

    # Get the top N indices for each English entity (row)
    top_n_indices = torch.topk(all_scores, N, dim=1, largest=True).indices

    # Check if the correct French entity is in the top N indices for each English entity
    hits = torch.any(top_n_indices == correct_indices.unsqueeze(1), dim=1)

    # Calculate the Hits@N as the ratio of correct hits
    hits_at_n = hits.float().mean().item()

    return hits_at_n


# Load model and data
_, _, test_loader = get_data_loaders(batch_size=32, num_neg_samples=0)
model = NodeAlignmentModel(384).to("cuda")
# Pick latest model
model.load_state_dict(torch.load("saved_models/model.pt", weights_only=True))

all_scores = get_scores(model, test_loader)

# Calculate Hits@k
k_values = [1, 3, 5, 10]
hits = {}
for n in k_values:
    hits[n] = calculate_hits_at_n(all_scores, n)
    print(f"Hit@{n:2}: {hits[n]:.8f}")
