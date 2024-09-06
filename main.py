import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import numpy as np
import json

# Load models 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_minilm = SentenceTransformer('all-MiniLM-L6-v2').to(device)
tokenizer_labse = AutoTokenizer.from_pretrained('sentence-transformers/LaBSE')
model_labse = AutoModel.from_pretrained('sentence-transformers/LaBSE').to(device)

# Mean Pooling function 
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Load data
with open('data/records_dbp15k_en_clean.json', 'r', encoding='utf-8-sig') as f:
    en_data = json.load(f)
with open('data/records_dbp15k_fr_clean.json', 'r', encoding='utf-8-sig') as f:
    fr_data = json.load(f)

# Load relationship data
with open('data/rel_dbp15k_en.json', 'r', encoding='utf-8-sig') as f:
    en_rel_data = json.load(f)
with open('data/rel_dbp15k_fr.json', 'r', encoding='utf-8-sig') as f:
    fr_rel_data = json.load(f)

# Create dictionaries for quick lookup
en_rel_dict = {item['source_uri']: item for item in en_rel_data}
fr_rel_dict = {item['source_uri']: item for item in fr_rel_data}

# Prepare sentences for property embeddings
def prepare_property_sentence(entry):
    uri = entry["n.uri"]
    properties = " ".join([f"{k}: {v}" for k, v in entry["properties(n)"].items()])
    return f"URI: {uri} Properties: {properties}"

# Prepare sentences for relationship embeddings
def prepare_relationship_sentence(rel_data):
    outgoing = " ".join([f"OUT {r['type']} {r['neighbor_uri']}" for r in rel_data.get('outgoing_relationships', [])])
    incoming = " ".join([f"IN {r['type']} {r['neighbor_uri']}" for r in rel_data.get('incoming_relationships', [])])
    return f"{outgoing} {incoming}"

en_property_sentences = [prepare_property_sentence(entry) for entry in en_data]
fr_property_sentences = [prepare_property_sentence(entry) for entry in fr_data]
en_relationship_sentences = [prepare_relationship_sentence(en_rel_dict.get(entry["n.uri"], {})) for entry in en_data]
fr_relationship_sentences = [prepare_relationship_sentence(fr_rel_dict.get(entry["n.uri"], {})) for entry in fr_data]

# Compute embeddings 
en_property_embeddings_minilm = model_minilm.encode(en_property_sentences, convert_to_tensor=True, device=device)
fr_property_embeddings_minilm = model_minilm.encode(fr_property_sentences, convert_to_tensor=True, device=device)

def compute_labse_embeddings(sentences):
    embeddings = []
    batch_size = 32
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        encoded_input = tokenizer_labse(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
        with torch.no_grad():
            model_output = model_labse(**encoded_input)
        batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings)

en_property_embeddings_labse = compute_labse_embeddings(en_property_sentences)
fr_property_embeddings_labse = compute_labse_embeddings(fr_property_sentences)

# Compute relationship embeddings
en_relationship_embeddings = model_minilm.encode(en_relationship_sentences, convert_to_tensor=True, device=device)
fr_relationship_embeddings = model_minilm.encode(fr_relationship_sentences, convert_to_tensor=True, device=device)

en_property_embeddings_minilm = torch.nn.functional.normalize(en_property_embeddings_minilm, p=2, dim=1)
fr_property_embeddings_minilm = torch.nn.functional.normalize(fr_property_embeddings_minilm, p=2, dim=1)

en_property_embeddings_labse = torch.nn.functional.normalize(en_property_embeddings_labse, p=2, dim=1)
fr_property_embeddings_labse = torch.nn.functional.normalize(fr_property_embeddings_labse, p=2, dim=1)

en_relationship_embeddings = torch.nn.functional.normalize(en_relationship_embeddings, p=2, dim=1)
fr_relationship_embeddings = torch.nn.functional.normalize(fr_relationship_embeddings, p=2, dim=1)

# Compute similarity matrices separately for property and relationship embeddings
property_similarity_matrix_minilm = torch.matmul(en_property_embeddings_minilm, fr_property_embeddings_minilm.T).cpu().numpy()
property_similarity_matrix_labse = torch.matmul(en_property_embeddings_labse, fr_property_embeddings_labse.T).cpu().numpy()
relationship_similarity_matrix = torch.matmul(en_relationship_embeddings, fr_relationship_embeddings.T).cpu().numpy()

# Combine property and relationship similarity matrices using the provided weights
combined_similarity_matrix = (0.3 * property_similarity_matrix_minilm +
                               0.5 * property_similarity_matrix_labse +
                               0.2 * relationship_similarity_matrix)

# Find the top 20 matches for each entity
top_k = 20
direct_matches = []
for i, en_entry in enumerate(en_data):
    similarity_scores = combined_similarity_matrix[i]
    top_indices = np.argsort(-similarity_scores)[:top_k]
    top_similar_nodes = [{"uri": fr_data[idx]["n.uri"], "score": float(similarity_scores[idx])} for idx in top_indices]
    direct_matches.append({"n.uri": en_entry["n.uri"], "topSimilarNodes": top_similar_nodes})

# Save the results
with open('Results/similar_entities_dbp15k_combined.json', 'w', encoding='utf-8-sig') as f:
    json.dump(direct_matches, f, indent=2)

