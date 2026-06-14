"""
Ranqueia as palavras do dataset (pastas com vídeo em data/raw_videos) pela
frequência de uso no português, usando a lista do OpenSubtitles PT-BR
(data/freq_pt_br_50k.txt). Serve para escolher só as N palavras mais usadas e
reduzir o número de classes do treino.

Normalização para casar com a lista:
  - minúsculas; '_' e '-' viram espaço; dígitos/pontuação removidos
    (ex: 'VISTA1' -> 'vista', 'LEITE-DE-COCO' -> 'leite de coco')
  - compostos: usa a frequência do token de CONTEÚDO mais raro (o menor count),
    o que naturalmente ignora palavras-função ('de', 'a') que são muito comuns.

Saída:
  - data/palavras_por_frequencia.csv  (todas, ordenadas por frequência)
  - resumo no terminal: quantas casaram e o ln(N) em vários cortes top-K.

Uso:
    python scripts/ranquear_palavras.py
"""
import csv
import os
import re
import math

RAIZ      = 'data/raw_videos'
FREQ_FILE = 'data/freq_pt_br_50k.txt'
SAIDA_CSV = 'data/palavras_por_frequencia.csv'


def carregar_frequencias(caminho):
    freq = {}
    with open(caminho, encoding='utf-8') as f:
        for linha in f:
            partes = linha.split()
            if len(partes) == 2:
                palavra, cont = partes
                freq[palavra] = int(cont)
    return freq


def normalizar(palavra):
    """Folder name -> lista de tokens minúsculos só com letras (mantém acentos)."""
    s = palavra.lower().replace('_', ' ').replace('-', ' ')
    s = re.sub(r'[^a-zà-ÿ ]', ' ', s)   # remove dígitos, parênteses, etc.
    return [t for t in s.split() if t]


def frequencia_da_palavra(palavra, freq):
    """Count de uso: token de conteúdo mais raro encontrado; 0 se nada casar."""
    tokens = normalizar(palavra)
    encontrados = [freq[t] for t in tokens if t in freq]
    return min(encontrados) if encontrados else 0


def coletar_palavras_com_video(raiz):
    """Retorna lista de (assunto, palavra) que têm ao menos 1 .mp4."""
    pares = []
    for assunto in sorted(os.listdir(raiz)):
        pa = os.path.join(raiz, assunto)
        if not os.path.isdir(pa):
            continue
        for palavra in sorted(os.listdir(pa)):
            pp = os.path.join(pa, palavra)
            if os.path.isdir(pp) and any(f.lower().endswith('.mp4') for f in os.listdir(pp)):
                pares.append((assunto, palavra))
    return pares


def main():
    freq  = carregar_frequencias(FREQ_FILE)
    pares = coletar_palavras_com_video(RAIZ)

    ranqueadas = []
    for assunto, palavra in pares:
        cont = frequencia_da_palavra(palavra, freq)
        ranqueadas.append((palavra, assunto, cont))
    ranqueadas.sort(key=lambda x: x[2], reverse=True)

    with open(SAIDA_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['rank', 'palavra', 'assunto', 'contagem', 'encontrada'])
        for i, (palavra, assunto, cont) in enumerate(ranqueadas, 1):
            w.writerow([i, palavra, assunto, cont, 'sim' if cont > 0 else 'nao'])

    total       = len(ranqueadas)
    encontradas = sum(1 for _, _, c in ranqueadas if c > 0)

    print(f"Palavras com vídeo : {total}")
    print(f"Casaram na lista   : {encontradas} ({encontradas / total:.0%})")
    print(f"Sem frequência (0) : {total - encontradas}  (raras/nomes próprios/letras)")
    print(f"CSV salvo em       : {SAIDA_CSV}\n")

    print("Se mantermos só as top-K mais usadas:")
    print(f"{'top-K':>7s} | {'classes reais':>13s} | {'ln(N)':>6s}")
    print('-' * 34)
    for k in [100, 200, 300, 500, 1000, 1500]:
        sub = [r for r in ranqueadas[:k] if r[2] > 0]   # só as que têm uso real
        n = len(sub)
        print(f"{k:7d} | {n:13d} | {math.log(n):6.2f}" if n else f"{k:7d} |      0 |   -")

    print("\nTop 20 palavras mais usadas no dataset:")
    for palavra, assunto, cont in ranqueadas[:20]:
        print(f"  {cont:>9d}  {palavra:<22s} [{assunto}]")


if __name__ == '__main__':
    main()
