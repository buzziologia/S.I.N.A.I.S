"""
Avalia um modelo treinado no hold-out de vídeos próprios
(data/meus_features_holdout/PALAVRA/*.npy).

É a métrica honesta do projeto: vídeos do sinalizante real, gravados em
takes que NUNCA entraram no treino (nem como augmentação).

Com --raw, avalia nos vídeos ORIGINAIS do INES (data/processed_features,
filtrados às classes do modelo). Atenção: esses vídeos normalmente fizeram
parte do treino — mede retenção do domínio INES, não generalização.

Uso:
    python scripts/avaliar_holdout.py --pesos models/saved_weights/XXX_modelo.pth
    python scripts/avaliar_holdout.py --pesos ... --raw
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

# Permite importar de models/ e reutilizar o pré-processamento do treino
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from models.lstm_classifier import LIBRASClassifier
from models.preprocess import (preparar_sequencia, colapsar_variante, parse_bases,
                               unificar_rotulos, INPUT_DIM)
from train import _construir_mapa_classes


def coletar_amostras_holdout(dir_holdout: str, classes: list) -> list:
    """Retorna [(palavra, caminho_npy)] do hold-out PALAVRA/*.npy."""
    amostras = []
    for palavra in sorted(os.listdir(dir_holdout)):
        pasta = os.path.join(dir_holdout, palavra)
        if not os.path.isdir(pasta):
            continue
        if palavra not in classes:
            print(f"[AVISO] '{palavra}' não está no vocabulário do modelo — ignorada.")
            continue
        for arq in sorted(os.listdir(pasta)):
            if arq.endswith(".npy"):
                amostras.append((palavra, os.path.join(pasta, arq)))
    return amostras


def coletar_amostras_raw(dir_features: str, dir_videos: str, classes: list) -> list:
    """
    Retorna [(palavra, caminho_npy)] dos vídeos originais do INES cujas
    palavras estão no vocabulário do modelo (mapa stem→palavra vem da
    estrutura raw_videos/ASSUNTO/PALAVRA/video.mp4).
    """
    mapa = _construir_mapa_classes(dir_videos)
    vocab = set(classes)
    amostras = []
    for arq in sorted(os.listdir(dir_features)):
        if not arq.endswith(".npy"):
            continue
        palavra = mapa.get(os.path.splitext(arq)[0])
        if palavra in vocab:
            amostras.append((palavra, os.path.join(dir_features, arq)))
    return amostras


def main():
    parser = argparse.ArgumentParser(description="Avalia um modelo no hold-out de vídeos próprios")
    parser.add_argument("--pesos", required=True,
                        help="Caminho do *_modelo.pth (o *_classes.json é inferido)")
    parser.add_argument("--dir_holdout", default="data/meus_features_holdout",
                        help="Pasta PALAVRA/*.npy com as features de hold-out")
    parser.add_argument("--raw", action="store_true",
                        help="Avalia nos vídeos originais do INES (processed_features) "
                             "em vez do hold-out — mede retenção, não generalização")
    parser.add_argument("--dir_raw", default="data/processed_features",
                        help="Pasta dos .npy originais do INES (com --raw)")
    parser.add_argument("--dir_videos", default="data/raw_videos",
                        help="Pasta dos vídeos originais, p/ mapear classes (com --raw)")
    parser.add_argument("--unificar", default="QUE",
                        help="Palavras cujas variantes numeradas são unificadas na "
                             "classificação (somando probabilidades). Separadas por "
                             "vírgula; '' desativa. Padrão: QUE")
    args = parser.parse_args()

    caminho_classes = args.pesos.replace("_modelo.pth", "_classes.json")
    with open(caminho_classes, encoding="utf-8") as f:
        classes = json.load(f)["classes"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = LIBRASClassifier(input_dim=INPUT_DIM, num_classes=len(classes)).to(device)
    modelo.load_state_dict(torch.load(args.pesos, map_location=device))
    modelo.eval()

    if args.raw:
        amostras = coletar_amostras_raw(args.dir_raw, args.dir_videos, classes)
        fonte = f"{args.dir_raw} (vídeos INES — estiveram no TREINO; mede retenção)"
    else:
        amostras = coletar_amostras_holdout(args.dir_holdout, classes)
        fonte = args.dir_holdout

    # Fusão de variantes na CLASSIFICAÇÃO: o modelo continua prevendo QUE1/QUE2,
    # mas as probabilidades são somadas e o acerto é medido pela palavra-base.
    bases = parse_bases(args.unificar)
    rotulos, mapa_idx = unificar_rotulos(classes, bases)
    if len(rotulos) < len(classes):
        print(f"[INFO] Variantes unificadas na classificação ({args.unificar}): "
              f"{len(classes)} classes → {len(rotulos)} rótulos.")

    acertos, total = 0, 0
    por_classe = defaultdict(lambda: [0, 0])   # rótulo -> [acertos, total]
    linhas = []                                 # relatório por arquivo

    for palavra, caminho in amostras:
        t = preparar_sequencia(np.load(caminho))
        with torch.no_grad():
            logits = modelo(torch.from_numpy(t).unsqueeze(0).to(device))
            probs = logits.softmax(1)[0].cpu().numpy()

        probs_u = np.zeros(len(rotulos))
        np.add.at(probs_u, mapa_idx, probs)
        ordem = probs_u.argsort()[::-1][:3]

        verdade = colapsar_variante(palavra, bases)
        pred = rotulos[ordem[0]]
        ok = pred == verdade

        total += 1
        por_classe[verdade][1] += 1
        if ok:
            acertos += 1
            por_classe[verdade][0] += 1

        arq = os.path.basename(caminho)
        desc_top3 = "  ".join(f"{rotulos[i]}({probs_u[i] * 100:.0f}%)" for i in ordem)
        linhas.append(f"  {'✓' if ok else '✗'} {verdade:<8} {arq:<22} → {desc_top3}")

    print(f"\nModelo: {args.pesos}  ({len(classes)} classes)")
    print(f"Avaliado em: {fonte}\n")
    for linha in linhas:
        print(linha)

    print("\nPor palavra:")
    for palavra in sorted(por_classe):
        a, n = por_classe[palavra]
        print(f"  {palavra:<8} {a}/{n}")

    if total == 0:
        print("\n[ERRO] Nenhuma amostra encontrada.")
        sys.exit(1)

    rotulo = "ACURÁCIA RAW (INES, visto no treino)" if args.raw else "ACURÁCIA HOLD-OUT"
    print(f"\n{rotulo}: {acertos}/{total} = {acertos / total * 100:.1f}%  "
          f"(acaso: {100 / len(rotulos):.1f}%)")


if __name__ == "__main__":
    main()
