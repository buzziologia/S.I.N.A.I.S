"""
Script: inspect_ines_vocabulary.py
Projeto: S.I.N.A.I.S - Tradutor de LIBRAS

Propósito:
    Analisar o vocabulário completo do dicionário INES, gerar estatísticas
    e exportar um CSV limpo com metadados de todas as palavras.
    Útil para selecionar quais classes baixar primeiro.
"""

import re
import json
import logging
import requests
import pandas as pd
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sinais.inspector")

URL_PALAVRAS_JS = "https://dicionario.ines.gov.br/public/site/js/palavras.js"
URL_ASSUNTOS_JS = "https://dicionario.ines.gov.br/public/site/js/assuntos.js"

HEADERS: dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (SINAIS-LIBRAS-Research/1.0; Educational Use)",
    "Referer": "https://dicionario.ines.gov.br/",
}

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"


def _fetch_js_as_json(url: str, var_name: str) -> list[dict[str, Any]]:
    """Busca e extrai uma variável JSON de um arquivo JavaScript do INES."""
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()
    pattern = rf"var\s+{re.escape(var_name)}\s*=\s*(\[.*?\]);"
    match = re.search(pattern, response.text, re.DOTALL)
    if not match:
        raise ValueError(f"Variável '{var_name}' não encontrada.")
    return json.loads(match.group(1))


def main() -> None:
    """Gera relatório e CSV de análise do vocabulário completo do INES."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Carregando vocabulário do INES...")
    palavras = _fetch_js_as_json(URL_PALAVRAS_JS, "palavras")
    assuntos = _fetch_js_as_json(URL_ASSUNTOS_JS, "assuntos")

    # Cria mapeamento de assunto id -> nome para o merge.
    assunto_map = {str(a["id"]): a["nome"] for a in assuntos}

    df = pd.DataFrame(palavras)
    df["assunto_nome"] = df["assunto"].astype(str).map(assunto_map).fillna("DESCONHECIDO")
    df["tem_video_mp4"] = df["video"].str.endswith(".mp4", na=False)

    log.info(f"\n{'='*50}")
    log.info(f"📊 VOCABULÁRIO INES — ESTATÍSTICAS")
    log.info(f"{'='*50}")
    log.info(f"Total de palavras          : {len(df)}")
    log.info(f"Com vídeo .mp4             : {df['tem_video_mp4'].sum()}")
    log.info(f"Sem vídeo ou vídeo antigo  : {(~df['tem_video_mp4']).sum()}")
    log.info(f"\nDistribuição por Assunto:")
    subject_stats = (
        df.groupby("assunto_nome")
        .agg(total=("id", "count"), com_mp4=("tem_video_mp4", "sum"))
        .sort_values("total", ascending=False)
    )
    log.info(f"\n{subject_stats.to_string()}")

    # Exporta CSV limpo para análise.
    csv_path = OUTPUT_DIR / "vocabulario_ines.csv"
    df[["ident", "palavra", "assunto_nome", "video", "tem_video_mp4", "descricao"]].to_csv(
        csv_path, index=False, encoding="utf-8-sig"
    )
    log.info(f"\n✅ CSV exportado: {csv_path}")


if __name__ == "__main__":
    main()
