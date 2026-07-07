# Atalhos de comandos do S.I.N.A.I.S  -  rode:  just <comando>
# Ex.:  just train --top_k 300 --epochs 50
#       just camera --pesos models/saved_weights/1781288399662_modelo.pth
#
# Precisa do programa `just` instalado (winget install --id Casey.Just).

# Python a usar. Defina SINAIS_PYTHON p/ apontar seu ambiente (ex: "python" com
# o conda ativado, ou o caminho completo do python.exe do env sinais311).
# Sem a variável, usa o caminho abaixo (máquina do Mateus).
python := env_var_or_default("SINAIS_PYTHON", "C:/Users/matbu/anaconda3/envs/sinais311/python.exe")

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

# DEMO: webcam com o melhor modelo atual (top-10, treinado com 3 sinalizantes - 98% no hold-out)
demo:
    & "{{python}}" scripts/testar_camera.py --pesos models\saved_weights\1783189508175_modelo.pth --confianca 0.3

# 5) Coleta seus próprios vídeos. Ex: just coletar --palavras CASA,AMOR --reps 10
coletar *args:
    & "{{python}}" scripts/training/coletar_videos.py {{args}}

# Baixa os vídeos originais do INES
download:
    & "{{python}}" scripts/utils/download_ines_videos.py

# Ranqueia as palavras por frequência de uso (gera data/palavras_por_frequencia.csv)
ranquear:
    & "{{python}}" scripts/utils/ranquear_palavras.py

# 6) Avalia um modelo no hold-out de vídeos próprios (ou --raw p/ vídeos INES)
avaliar *args:
    & "{{python}}" scripts/avaliar_holdout.py {{args}}
