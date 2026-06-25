# Atalhos de comandos do S.I.N.A.I.S  —  rode:  just <comando>
# Ex.:  just train --top_k 300 --epochs 50
#       just camera --pesos models/saved_weights/1781288399662_modelo.pth
#
# Precisa do programa `just` instalado (winget install --id Casey.Just).

# Python do ambiente conda (não precisa ativar o env)
python := "C:/Users/matbu/anaconda3/envs/sinais311/python.exe"

# UTF-8 em toda saída (acentos, ✓, ─ etc.)
export PYTHONUTF8 := "1"

# No Windows, roda as receitas no PowerShell
set windows-shell := ["powershell.exe", "-NoProfile", "-Command"]

# Sempre roda a partir da raiz do projeto (mesmo chamando `just` de uma subpasta).
# '.' é relativo ao diretório do justfile → a raiz do projeto. (É também o padrão do just.)
set working-directory := '.'

# Lista os comandos disponíveis (default quando roda `just` sem argumento)
default:
    @just --list

# 1) Extrai features dos vídeos: raw_videos -> processed_features (126-d, 2 mãos)
extract:
    & "{{python}}" scripts/training/extract_features.py

# 2) Augmentation offline (N variações por .npy). Ex: just augment 30
augment n="30":
    & "{{python}}" scripts/utils/augmentar_offline.py --n {{n}}

# 3) Treino (por palavra). Ex: just train --top_k 300 --epochs 50
train *args:
    & "{{python}}" scripts/train.py {{args}}

# 4) Reconhecimento pela webcam. Ex: just camera --pesos models/saved_weights/XXXX_modelo.pth
camera *args:
    & "{{python}}" scripts/testar_camera.py {{args}}

# Atalho: webcam com o modelo top-300 já fixado
camera-300:
    & "{{python}}" scripts/testar_camera.py --pesos models\saved_weights\1781288399662_modelo.pth --confianca 0.1 --sem_ood

# 5) Coleta seus próprios vídeos. Ex: just coletar --palavras CASA,AMOR --reps 10
coletar *args:
    & "{{python}}" scripts/training/coletar_videos.py {{args}}

# Baixa os vídeos originais do INES
download:
    & "{{python}}" scripts/utils/download_ines_videos.py

# Diagnóstico do pipeline (4 testes)
debug:
    & "{{python}}" scripts/utils/debug_treino.py

# Ranqueia as palavras por frequência de uso (gera data/palavras_por_frequencia.csv)
ranquear:
    & "{{python}}" scripts/utils/ranquear_palavras.py

# Experimentos
exp-grupos:
    & "{{python}}" scripts/experiments/experimento_grupos.py

exp-config:
    & "{{python}}" scripts/experiments/experimento_config_mao.py
