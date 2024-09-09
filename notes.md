# Experiments notes
Author: idriss

## Experiement 1:
I reworked the texts in the sentence preparations.
Example:

```md
Node name: Waldemar Pawlak
Properties:
	- office: Member of the Sejm.
	- president: Bogdan Borusewicz.
	- termStart: 1993-10-26.
	- constituency: 16.
	- imagesize: 240.
	- uri: http://dbpedia.org/resource/Waldemar_Pawlak.
	- termEnd: 2012-11-27.
	- title: Leader of the Polish People's Party.
	- spouse: Elżbieta Pawlak.
	- awards: 40.
	- name: Waldemar Pawlak.
	- birthDate: 1959-09-05.
    - years: 1992.
```

```md
Outgoing relations:
	type: title, value: List of Spanish monarchs.
	type: title, value: Prince of Asturias.
	type: issue, value: Isabel II of Spain.
	type: deathPlace, value: Madrid.
	type: predecessor, value: Charles IV of Spain.
	type: predecessor, value: Joseph Bonaparte.
	type: successor, value: Isabel II of Spain.
	type: successor, value: Joseph Bonaparte.
	type: father, value: Charles IV of Spain.
	type: house, value: House of Bourbon.
	type: placeOfBurial, value: El Escorial.
Incoming relations:
	type: leader, value: Captaincy General of Chile.
	type: leader, value: Viceroyalty of Peru.
	type: predecessor, value: Joseph Bonaparte.
	type: predecessor, value: Isabel II of Spain.
	type: spouse, value: Maria Christina of the Two Sicilies.
	type: successor, value: Charles IV of Spain.
	type: successor, value: Joseph Bonaparte.
	type: father, value: Infanta Luisa Fernanda, Duchess of Montpensier.
	type: father, value: Isabel II of Spain.
	type: monarch, value: Francisco Javier Venegas.
	type: commander, value: Peruvian War of Independence.
```
### Results
Hits@1: 0.7406666666666667
Hits@5: 0.8668666666666667
Hits@10: 0.9014666666666666
Hits@20: 0.9294
Total nodes: 15000


Difference from orginal (@Oumida)

Hits@1: -0.101
Hits@5: -0.069
Hits@10: -0.054
Hits@20: -0.041

### Results with LABSE
Hits@1: 0.8454
Hits@5: 0.9321333333333334
Hits@10: 0.9532
Hits@20: 0.9681333333333333
Total nodes: 15000


Difference from orginal (@Oumida)

Hits@1: 0.004
Hits@5: -0.003
Hits@10: -0.002
Hits@20: -0.003

## Experiement 2:
For each entity, I calculated the embeddings of its name, attributes, values, and relationships.
Then I stored them in pkl files along with a meta data file containing the entity name and indices of the attributes, values, and relationships.

In the script `interaction_model.py`, I calculated the distance between the entities in english and french.
Then I showed topk closest. After that I measured hit@n for n = 1, 3, 5, 10 with entities embeddings only.

Measuring hit@n...
hit@1:  0.7976666666666666
hit@3:  0.8538666666666667
hit@5:  0.8706666666666667
hit@10:  0.8892


## Experiement 3:
`entity_alignment_2.py`
- Concatenating all vectors of an entity with the average of the vectors of its attributes, values, and relationships.
Hit@K Metrics:
Hit@1: 0.8312
Hit@2: 0.8683
Hit@3: 0.8833
Hit@5: 0.8979
Hit@10: 0.9139

- Using max instead of average
Hit@K Metrics:
Hit@1: 0.8293
Hit@2: 0.8659
Hit@3: 0.8816
Hit@5: 0.8961
Hit@10: 0.9144

- Removing attributes, values, and relationships from the stacked vector we get the same results as experiment 2.
Hit@K Metrics:
Hit@1: 0.7977
Hit@2: 0.8365
Hit@3: 0.8539
Hit@5: 0.8707
Hit@10: 0.8892

- Add average of neighbors entities embedding to the stacked vector
Hit@K Metrics:
Hit@1: 0.8392
Hit@3: 0.8906
Hit@5: 0.9059
Hit@10: 0.9238
