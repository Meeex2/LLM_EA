import torch
from torch import nn
from torch.optim.adam import Adam
from tqdm import tqdm

from dataset import get_data_loaders
from node_alignement_model import NodeAlignmentModel


def main():
    # Instantiate the model
    embed_dim = 384
    num_epochs = 150
    num_neg_samples = 20
    margin = 1

    # Optimizer and loss function
    model = NodeAlignmentModel(embed_dim).to("cuda")
    optimizer = Adam(model.parameters(), lr=0.0005)
    loss_func = nn.MarginRankingLoss(margin)

    # Load dataset
    train_loader, val_loader, _ = get_data_loaders(
        batch_size=512, num_neg_samples=num_neg_samples
    )

    starting_training = r"""
   _____ __             __  _                __             _       _                __
  / ___// /_____ ______/ /_(_)___  ____ _   / /__________ _(_)___  (_)___  ____ _   / /
  \__ \/ __/ __ `/ ___/ __/ / __ \/ __ `/  / __/ ___/ __ `/ / __ \/ / __ \/ __ `/  / / 
 ___/ / /_/ /_/ / /  / /_/ / / / / /_/ /  / /_/ /  / /_/ / / / / / / / / / /_/ /  /_/  
/____/\__/\__,_/_/   \__/_/_/ /_/\__, /   \__/_/   \__,_/_/_/ /_/_/_/ /_/\__, /  (_)   

    """
    print(starting_training)
    for epoch in range(num_epochs):
        model.train()  # Ensure the model is in training mode
        total_loss = 0
        for batch in tqdm(train_loader):
            # Zero out the gradients
            optimizer.zero_grad()

            # Move input tensors to CUDA
            e1 = batch["e1"].to("cuda")
            e2 = batch["e2"].to("cuda")

            for i in range(num_neg_samples):
                neg_e2 = batch[f"neg_e2_{i}"].to("cuda")
                p1 = model(e1, e2)
                p2 = model(e1, neg_e2)

                # Compute loss
                loss = loss_func(p1, p2, torch.ones_like(p1))

                # Backpropagation and optimization step
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        avg_loss = total_loss / (len(train_loader) * num_neg_samples)
        print(f"Epoch {epoch+1:2d}/{num_epochs}, Training Loss  : {avg_loss:.8f}")

        # Validation phase (no gradient updates)
        model.eval()
        val_loss = 0.0
        mean_p_pos = -1
        mean_p_neg = -1
        with torch.no_grad():
            for batch in val_loader:
                # Move input tensors to CUDA
                e1 = batch["e1"].to("cuda")
                e2 = batch["e2"].to("cuda")
                # Forward pass through the model for both English and French node pairs

                for i in range(num_neg_samples):
                    neg_e2 = batch[f"neg_e2_{i}"].to("cuda")
                    p1 = model(e1, e2)
                    p2 = model(e1, neg_e2)
                    neg_e2 = neg_e2.to("cuda")

                    # Forward pass through the model for both English and French node pairs
                    p1 = model(e1, e2)
                    mean_p_pos = p1.mean().item()

                    p2 = model(e1, neg_e2)
                    mean_p_neg = p2.mean().item()

                    # Compute loss
                    val_loss += loss_func(p1, p2, torch.ones_like(p1)).item()
            print(
                f"Epoch {epoch+1:2d}/{num_epochs}, Validation Loss: {
                val_loss/(num_neg_samples*len(val_loader)):.8f
                }"
            )
            print("================================================================")
            print("mean_p_pos: ", mean_p_pos)
            print("mean_p_neg: ", mean_p_neg)
            print("================================================================")

    # Save the model

    torch.save(model.state_dict(), "saved_models/model.pt")


if __name__ == "__main__":
    main()
