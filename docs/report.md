# Report
## Methodology
In all experiements we follow a general approach. It consists of generating a textual description of the entity in some form.
Then we use an embedding model to encode the textual descriptions. In each experiemtn we try a different way to write the textual descriptions. We chose to use the model all-MiniLM-L6-v2 from SentenceTransformers.

There were two approaches that were followed. The first is updating the embedding given by All-MiniLM-L6-v2 with the help of a neural network. The second is training a neural network to predict directly the similarity of two entities.

## Experiment
In this experiment, we focused on improving entity alignment by experimenting with various ways of combining entity embeddings with their attributes, values, and relationships. We explored multiple approaches for constructing the final embeddings and evaluated their performance using `Hit@K` metrics.

### 1. **Concatenating Entity, Attributes, Values, and Relationships**  
In the first approach, we concatenated the embedding of each entity with the average of the embeddings of its attributes, values, and relationships. This method resulted in the following `Hit@K` metrics:

- **Hit@1:** 0.8312  
- **Hit@2:** 0.8683  
- **Hit@3:** 0.8833  
- **Hit@5:** 0.8979  
- **Hit@10:** 0.9139  

### 2. **Using Max Instead of Average for Aggregation**  
Next, we replaced the averaging operation with the max operation when aggregating the embeddings of attributes, values, and relationships. The performance slightly decreased:

- **Hit@1:** 0.8293  
- **Hit@2:** 0.8659  
- **Hit@3:** 0.8816  
- **Hit@5:** 0.8961  
- **Hit@10:** 0.9144  

### 3. **Removing Attributes, Values, and Relationships**  
In this case, we removed the embeddings of attributes, values, and relationships from the concatenated vector and only used the entity embeddings. This produced the same results as Experiment 1:

- **Hit@1:** 0.7977  
- **Hit@2:** 0.8365  
- **Hit@3:** 0.8539  
- **Hit@5:** 0.8707  
- **Hit@10:** 0.8892  

### 4. **Adding Average of Neighboring Entities**  
Finally, we added the average embeddings of neighboring entities to the concatenated vector, further enriching the entity representation. This improved the overall performance:

- **Hit@1:** 0.8392  
- **Hit@3:** 0.8906  
- **Hit@5:** 0.9059  
- **Hit@10:** 0.9238  

In conclusion, the inclusion of neighboring entities in the stacked vector led to the best performance, suggesting that leveraging local graph structure improves entity alignment.

## V2
## Experiment 2:
First, we generate embeddings for each entity along with its attributes, values, and relationships. In this experiment, we explored different strategies to combine these embeddings to improve entity alignment performance. We investigated four main approaches, each designed to enhance alignment accuracy. The performance of each method is evaluated using Hit@K metrics.


### 1. Concatenating Entity with the Average of Attributes, Values, and Relationships
In the first method, we concatenate the entity's embedding with the average of its attributes, values, and relationships embeddings. This combination attempts to encapsulate a holistic representation of each entity and its associated metadata.

### 2. Using Max Aggregation Instead of Average
For the second approach, we replace the averaging step with a max operation, where the maximum value of each embedding dimension is taken across the attributes, values, and relationships. The goal is to capture the strongest feature representations rather than their mean.

### 3. Using Only Entity Embeddings
In the third approach, we omit the embeddings of attributes, values, and relationships, using only the entity embeddings themselves. This serves as a control to measure the impact of removing additional metadata from the vector representation.

### 4. Adding Neighbor Entities to the Stacked Vector
In the final approach, we concatenate the average embeddings of neighboring entities to the entity vector. By incorporating information about neighboring entities, we aim to capture richer contextual information within the knowledge graph structure.

### Results:
The following table presents the Hit@K metrics for each method:

| Method                                                 | Hit@1  | Hit@5  | Hit@10 | Hit@20 |
|--------------------------------------------------------|--------|--------|--------|--------|
| Concat + Avg of Attributes, Values, and Relationships  | 0.8323 | 0.8991 | 0.9152 | 0.9306 |
| Concat + Max of Attributes, Values, and Relationships  | 0.8293 | 0.8961 | 0.9144 | 0.9291 |
| Entity Embeddings Only                                 | 0.7977 | 0.8707 | 0.8892 | 0.9056 |
| Concat + Avg of Neighboring Entities                   | 0.8389 | 0.9051 | 0.9205 | 0.9360 |


