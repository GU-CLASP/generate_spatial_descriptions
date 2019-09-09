### Abstract

Generating grounded image descriptions requires associating linguistic units with their corresponding visual clues.
A common method is to train a decoder language model with attention mechanism over convolutional visual features. % when generating descriptions.
Attention weights align 
the stratified visual features 
arranged by their location with tokens, most commonly words, in the target description.
However, 
words such as 
spatial relations (e.g. *next to* and *under*) are not directly referring to geometric arrangements of pixels but to complex geometric and conceptual representations.
The aim of this paper is to evaluate what representations facilitate generating image descriptions with spatial relations and lead to better grounded language generation. 
In particular, we investigate the contribution of four different representational modalities in generating relational referring expressions:
(i) (pre-trained) convolutional visual features, (ii) spatial attention over visual features, (iii) top-down geometric relational knowledge between objects, and (iv) world knowledge captured by contextual embeddings in language models.

- Evaluations
- Live demo (based on mobilenet)
