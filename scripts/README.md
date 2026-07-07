# 📜 scripts/ - Scripts do Pipeline S.I.N.A.I.S

Scripts organizados por fase do pipeline. O README da raiz descreve o fluxo completo; aqui vai o mapa rápido de cada arquivo.

| Script | Fase | O que faz |
|---|---|---|
| `utils/download_ines_videos.py` | 1. Coleta INES | Baixa vídeo + imagens por palavra do Dicionário INES para `data/raw_videos/{ASSUNTO}/{PALAVRA}/`. Retomada automática via `data/download_log.csv`. |
| `training/extract_features.py` | 2. Features | Converte cada `.mp4` em matriz `(N, 126)` (`data/processed_features/`). Pula existentes. |
| `utils/ranquear_palavras.py` | 2b. Vocabulário | Ranqueia as palavras por frequência de uso PT-BR → `data/palavras_por_frequencia.csv` (usado por `--top_k`). |
| `utils/augmentar_offline.py` | 2c. Augmentação | Gera N variações por `.npy` em `data/augmented_features/` (necessário com 1 vídeo/palavra). |
| `training/coletar_videos.py` | 3. Coleta própria | Grava vídeos seus pela webcam (referência INES na tela, revisão por take, `--holdout` separa reps de teste). |
| `train.py` | 4. Treino | Treina a LSTM. `--top_k K` filtra vocabulário; `--dir_meus data/meus_features` inclui vídeos próprios. |
| `avaliar_holdout.py` | 5. Avaliação | Acurácia no hold-out de vídeos próprios (métrica honesta) ou nos vídeos INES (`--raw`). |
| `testar_camera.py` | 6. Ao vivo | Reconhecimento pela webcam com rejeição OOD (KNN, cache em disco) e vídeo de referência. |
| `experiments/README.md` | - | Registro de experimentos já concluídos e seus resultados (não re-investigar). |

O pré-processamento das sequências (recorte de atividade, padding, normalização, augmentação) é **um módulo único**: `models/preprocess.py`. Treino, avaliação e câmera importam dele - nunca duplique essas funções em um script.
