# AweDist - Attention-aware New Input Embeddings

Code for our preprint "AweDist: Attention-aware Embedding Distillation for New Input Token Embeddings".

Paper on arXiv: https://arxiv.org/abs/2505.20133.

![Awedist method illustration](assets/awedist.png)

## Setup

We use [`uv`](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) for dependency management. You can either install our dependencies in a local `.venv/` via:

```sh
uv sync --frozen --no-install-package flash-attn && uv sync --frozen
```

The two-step approach is necessary to correctly install `flash-attn` (you can also skip the second step).
Alternatively, you can use our pre-built Docker image at `konstantinjdobler/awedist:v0`.

## Running AweDist

As an example, to initialize input embeddings for new biomedical domain tokens for `meta-llama/Llama-3.1-8B-Instruct`, you can run:

```bash
python apply_embedding_init.py --model_path=meta-llama/Llama-3.1-8B-Instruct \
    --out_path="./Llama-3.1-8B-Instruct-Biomed" \
    --build_target_method=ner \
    --new_tokens_source="./new_tokens/combined_all_medical_word_counts.jsonl" \
    --dataset_path="ncbi/pubmed" \
    --init_method="awedist" \
    --awedist_lr="9e-5"
```

We generally find that learning rates around `1e-4` work well for many models and a broad range of learning rates are able to yield good results, however this might vary depending on your use case.
For further parameters, such as the `target_layer` and different `loss_method` and `init_method` options, please consult `apply_embedding_init.py` and our paper!

### Evaluation

Coming soon.

## Citation

```bibtex
@misc{dobler2025awedist,
    title={AweDist: Attention-aware Embedding Distillation for New Input Token Embeddings},
    author={Konstantin Dobler and Desmond Elliott and Gerard de Melo},
    year={2025},
    eprint={2505.20133},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## Acknowledgements

We thank the German Federal Ministry for Education and Research (BMBF) for their compute grant through the project "KI-Servicezentrum Berlin Brandenburg"(01IS22092). Konstantin Dobler further thanks the European Laboratory for Learning and Intelligent Systems (ELLIS) PhD program for support.
The research was supported by a research grant (VIL53122) from VILLUM FONDEN, and by the European Union’s Horizon 2020 research and innovation program under grant agreement No. 101135671 (TrustLLM).
