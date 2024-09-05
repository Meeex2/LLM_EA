# Entity Alignment for Graphs using LLMs

This repository contains the implementation of an entity alignment approach for knowledge graphs using Large Language Models (LLMs). The project includes code for data preprocessing, encoding nodes using various embedding methods, and evaluating the results. The specific focus is on the DBP15k dataset, a well-known benchmark for entity alignment tasks, with French and English language graphs. Other State of the art datasets are available [here](https://paperswithcode.com/task/entity-alignment)

## Environment Setup

### Install Conda
To manage dependencies efficiently, we use Conda. If you don't have Conda installed, follow these steps to install it:

1. Download the Conda installer from the official [Conda website](https://docs.conda.io/en/latest/miniconda.html).
2. Follow the installation instructions for your operating system.

### Create and Activate the Environment

Once Conda is installed, you can create a dedicated environment for this project using the `environment.yml` file provided in the repository.

1. Clone the repository:
   ```bash
   git clone https://github.com/Meeex2/LLM_EA.git
   cd your-repository-name
   ```

2. Create the environment:
   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:
   ```bash
   conda activate nlp_env
   ```

This will set up all the necessary dependencies and libraries needed to run the project.

## Project Overview

This project addresses the challenge of aligning equivalent entities across different knowledge graphs. By leveraging LLMs, we encode nodes in the graphs using various embeddings and then evaluate the alignment between entities.

### Repository Structure

- **main.py**: 
  - This is the core script where the encoding of nodes is performed. Different embedding methods are utilized to generate node embeddings, which are then used to calculate similarities and identify potential matches.

- **clean_data.py**: 
  - This script is responsible for preprocessing and cleaning the JSON data files. It handles inconsistencies in the data, ensuring that the information used for entity alignment is accurate and consistent.

- **data/**:
  - This folder contains the extracted features of the DBP15k dataset for both the English and French graphs. The files include:
    - `rel_dbp15k_fr.json`: Relationship data for the French graph.
    - `rel_dbp15k_en.json`: Relationship data for the English graph.
    - `records_dbp15k_fr.json`: Entity data for the French graph.
    - `records_dbp15k_fr_clean.json`: Cleaned entity data for the French graph.
    - `records_dbp15k_en.json`: Entity data for the English graph.
    - `records_dbp15k_en_clean.json`: Cleaned entity data for the English graph.

- **results/**:
  - This folder contains the results of the entity alignment process. It includes the top 10 candidate matches for each node, evaluated using different embedding methods.

## Running the Code

To execute the entity alignment process, follow these steps:

1. **Data Cleaning**: 
   - Running `clean_data.py` to clean the raw JSON data files. This step has already been done for you.
     ```bash
     python clean_data.py
     ```

2. **Node Encoding and Alignment**:
   - Run `main.py` to encode the nodes and perform the alignment.
     ```bash
     python main.py
     ```

3. **Results**:
   - The results will be saved in the `results/` folder, where you can review the top candidate matches for each node and calculate the Hits (SOTA metric).

## Conclusion

This project demonstrates an advanced method for aligning entities across knowledge graphs by leveraging the power of LLMs and various embedding techniques. By following the steps outlined in this README, you can reproduce the results and experiment with different embedding methods to improve entity alignment performance.

