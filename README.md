# DeepHFFT-m7G
DeepHFFT-m7G: A Dual-Channel Self-Attention and Hybrid Feature Fusion Framework for RNA m7G Modification Identification

## Graphical abstract
![last250405_Graphical abstract](https://github.com/user-attachments/assets/60f3d755-e6b4-4597-85a5-bef06dd2f36e)
## Flowchart of DeepHFFT-m7G
![last250324](https://github.com/user-attachments/assets/473fbc43-bf63-4305-ab1a-7de4214b40a9)
## RUN

```python
python create_dna2vec.py
```

```python
python create_freature.py
```

```python
python create_graph_dataset.py -k 25
```

```python
python train2.py -k 25 -e 100
```

