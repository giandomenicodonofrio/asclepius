# ECG Arrhythmia Classification

Pipeline completa per classificare aritmie ECG multiclasse con modelli ensemble e uno studio di ablation su configurazioni di derivazioni, preprocessing e data augmentation. Include script di training/valutazione, ricerca iperparametri, bootstrap per confronti statistici e notebook di analisi risultati.

## Struttura
- `src/config.py`: esperimenti e configurazioni di derivazioni condivise.
- `src/models/`: modelli ensemble e varianti.
- `src/training/`: loop di training/eval e utility (train/val/test split).
- `src/pipelines/`: entrypoint per ablation (train/eval) e hyperband parameter search.
- `src/scripts/`: utility (bootstrap, reference eval, dataset creation).
- `src/utils/`: data loader/generator, preprocessing, augmentation, metriche, visualizzazione.
- `notebooks/`: analisi risultati (es. `show_result.ipynb`).
- `resources/`: dati di esempio/sintetici e grafici esportati.
- `utils/`: shim di compatibilità per import legacy.

## Requisiti
- Python 3.10+ consigliato.
- TensorFlow/Keras, NumPy, h5py, tqdm, matplotlib, ecc. (se presente un `requirements.txt`, usalo per installare).
- GPU opzionale; il codice limita la memoria GPU a ~8GB per sicurezza.

## Dataset
- Formato atteso: file HDF5 con `X` (shape `[N, T, 12]`) e `y` (one-hot per 9 classi).
- Indici per classe: gruppi `x/{1..9}` contenenti gli indici per split stratificati.
- Percorso di default: `dataset` (override con `--dataset-path`).

## Come iniziare
Installazione (esempio):
```bash
python -m venv .venv && source .venv/bin/activate  # (Linux/Mac)
# su Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Valutazione del modello di riferimento
```bash
python -m src.scripts.reference_evaluation \
  --dataset-path dataset \
  --output-path reference_model_evaluation
```

### Ablation study (train o eval)
```bash
# Eval (default)
python -m src.pipelines.ablation_study \
  --dataset-path dataset \
  --output-path ablation_study \
  --mode evaluate

# Train (wrapper compatibile)
python -m src.pipelines.ablation_study_training_parallel \
  --dataset-path dataset \
  --output-path ablation_study_parallel
```

### Bootstrap per confronti tra modelli
```bash
python -m src.scripts.bootstrap_evaluation
# carica i risultati in ablation_study/evaluate_res e salva statistiche in bootstrap_result/
```

### Ricerca iperparametri (HyperBand)
```bash
python -m src.pipelines.parameter_search \
  --dataset-path dataset_72000_onehot_multiclass_id_fold
```

### Utility varie
- Generazione ECG sintetico: `python generate_synthetic_ecg.py`
- Confronto segnale/raw preprocessato: `python show_sample.py`

## Output e risultati
- Metriche per fold: `*/fold_{i}_score.pickle`
- Predizioni: `*/fold_{i}_y_test_y_preds.pickle`
- Challenge score: `*/challenge_score/fold{i}_score.txt`
- Checkpoint: `*/cp.ckpt*`
(Questi percorsi sono ignorati in git per non sporcare il repo.)

## Note e scelte progettuali
- Configurazioni di derivazioni e lista esperimenti sono centralizzate in `src/config.py`.
- Import uniformati a `src.*`; il package `utils/` rimane per compatibilità con codice/notebook legacy.
- Il bootstrap usa tutti gli esperimenti per cui esistono risultati su disco (`ablation_study/evaluate_res`).

