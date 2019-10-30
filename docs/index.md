# What Goes Into A Word: Generating Image Descriptions With Top-Down Spatial Knowledge
**Authors:** *Mehdi Ghanimifard and Simon Dobnik*

### Abstract

Generating grounded image descriptions requires associating linguistic units with their corresponding visual clues.
A common method is to train a decoder language model with attention mechanism over convolutional visual features.
Attention weights align
the stratified visual features
arranged by their location with tokens, most commonly words, in the target description.
However,
words such as
spatial relations (e.g. *next to* and *under*) are not directly referring to geometric arrangements of pixels but to complex geometric and conceptual representations.
The aim of this paper is to evaluate what representations facilitate generating image descriptions with spatial relations and lead to better grounded language generation.
In particular, we investigate the contribution of four different representational modalities in generating relational referring expressions:
(i) (pre-trained) convolutional visual features, (ii) spatial attention over visual features, (iii) top-down geometric relational knowledge between objects, and (iv) world knowledge captured by contextual embeddings in language models.


[[Paper](https://www.inlg2019.com/assets/papers/143_Paper.pdf)]
[[Slides](https://gu-clasp.github.io/generate_spatial_descriptions/presentation.pdf)]
[[Codes](https://github.com/GU-CLASP/generate_spatial_descriptions/tree/master/codes)]
[[Demo](https://gu-clasp.github.io/generate_spatial_descriptions/demo/)]

```
@inproceedings{ghanimifard-dobnik-2019-what,
    title = "What Goes Into A Word: Generating Image Descriptions With Top-Down Spatial Knowledge",
    author = "Ghanimifard, Mehdi. and Dobnik, Simon",
    booktitle = "Proceedings of the 12th International Conference on Natural Language Generation (INLG-2019)",
    month = "Oct",
    year = "2019",
    address = "Tokyo, Japan",
    publisher = "Association for Computational Linguistics",
    url = "",
    doi = "",
    pages = "",
}
```
