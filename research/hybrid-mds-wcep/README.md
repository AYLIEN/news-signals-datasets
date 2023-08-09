## Hybrid BART-based Multi-Document Summarization


This project packages a BART-based MDS model that was fine-tuned on extractive summaries from WCEP clusters. <br>
At test time, we first run extractive summarization and apply the model to that, which we limit to 5 sentences. <br>
The point of this approach is to limit the context size for more efficient inference. <br>


Install:

```
make dev
```

Download model:

```
make download_model
```

Run example (works on CPU):

```
make example
```
