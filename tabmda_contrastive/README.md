# TabMDA Contrastive Extension

The `tabmda_contrastive` package adds a contrastive-learning objective on top of
frozen TabPFN embeddings, following the design in
`TabMDA_Contrastive_Loss_Design_and_Codex_Task.docx`.

## What it does

1. **Freeze** the vendored TabPFN v1 encoder (`requires_grad=False`, `eval`).
2. For each original training sample, sample `K` labeled contexts (stratified,
   query-self excluded) and encode each `(query, context)` pair to obtain a
   `[N, K, D]` tensor of embeddings (D = 512 for the vendored encoder).
3. Train a small **MLP adapter + classification head** on the flattened
   `M = N * K` views, minimizing

   ```
   L = L_CE + λ_con * L_con + λ_ctx * L_ctx
   ```

   where `L_con` is a **grouped weighted SupCon** over the views:
   - *instance positives* (same record, different context) get total mass `α`,
   - *class positives* (different record, same label) get total mass `1 - α`,
   - *negatives* (different label) contribute only to the denominator.
   `L_ctx` is an optional cross-context consistency term (disabled by default).

4. Inference uses a **fixed protocol**: `K_eval` contexts per query, classified
   individually and averaged (`probability_mean`; `embedding_mean` is available
   as an alternative).

## Layout

```
tabmda_contrastive/
  config.py             ContrastiveConfig (defaults)
  context_sampler.py    leak-free context sampling (no self-context)
  embedding_backend.py  FrozenTabPFNBackend + IdentityBackend + cache
  losses.py             grouped weighted SupCon + CE + consistency
  model.py              Adapter + ContrastiveModel (r / z / logits)
  train.py              MinibatchSampler + training loop
  evaluate.py           fixed-protocol evaluation / prediction
  run.py                CLI (smoke / train / predict)
tests/                  acceptance tests (pytest)
```

## Environment

The vendored TabPFN v1 encoder is only importable from the `tabmda` conda env.
Run everything with `E:\miniconda3\envs\tabmda\python.exe` (or after
`conda activate tabmda`).

## Commands

### 1. Synthetic-data CPU smoke test (no TabPFN)

```bash
python -m tabmda_contrastive.run smoke --epochs 5 --n 200
```

Runs the full embed → train → fixed-protocol-eval pipeline on a synthetic
Gaussian-mixture dataset using the deterministic `identity` backend.

### 2. Real TabPFN single-seed experiment

```bash
python -m tabmda_contrastive.run train \
    --dataset vehicle --num_real_samples 100 --repeat_id 0 \
    --epochs 30 --cache_dir results_contrastive/cache
```

Supported datasets: `qsar-biodeg`, `vehicle`, `texture`,
`steel-plates-fault`, `MiceProtein`, `mfeat-fourier` (train split comes from
`data/index/<dataset>/split_<repeat_id>`; a dataset may reject unsupported
`--num_real_samples`).

### 3. Multi-seed ablation

```bash
python -m tabmda_contrastive.run train \
    --dataset vehicle --num_real_samples 100 --repeat_id 0 \
    --epochs 30 --seeds 0,1,2,3,4
```

### 4. Predict unlabeled rows from a saved checkpoint

```bash
python -m tabmda_contrastive.run predict \
    --checkpoint adapter_checkpoint.pt \
    --dataset vehicle --num_real_samples 100 --repeat_id 0
```

Predictions are written to `results_contrastive/predictions.npz`
(`probs` and `preds`). Needing a checkpoint named `adapter_checkpoint.pt` in the
working directory happens because `train()` currently saves to that fixed path;
you can also reload any checkpoint produced by `train()` elsewhere.

### 5. Run the test suite

```bash
python -m pytest tests/ -q
```

## Key configuration knobs

| option | default | meaning |
|---|---|---|
| `--K` | 2 | contexts per original sample at training |
| `--context_size` | None | None=whole pool, int=abs size, float=proportion |
| `--adapter_out_dim` | 128 | adapter / z / r representation dim |
| `--temperature` | 0.2 | SupCon temperature |
| `--alpha` | 0.5 | mass split instance- vs class-positives |
| `--lambda_con` | 0.1 | contrastive weight |
| `--lambda_ctx` | 0.0 | cross-context consistency weight (disabled) |
| `--contrastive_mode` | grouped | `grouped` (proposed) or `uniform` (ablation) |
| `--K_eval` | 2 | contexts per query at inference |
| `--aggregation` | probability_mean | `probability_mean` or `embedding_mean` |