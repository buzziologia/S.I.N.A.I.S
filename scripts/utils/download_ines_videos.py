"""
Script: download_ines_videos.py
Projeto: S.I.N.A.I.S - Tradutor de LIBRAS

Propósito:
    Baixar os 3 artefatos por palavra do Dicionário Digital do INES:
      1. Vídeo (.mp4)         - execução do sinal em LIBRAS
      2. Imagem do sinal (.jpg) - foto estática do sinal
      3. Imagem da mão (.jpg) - configuração da mão inicial (Classification cue)

Lógica (Passo a Passo):
    1. Buscar `palavras.js`, `assuntos.js` e `mao.js` e extrair os JSONs.
    2. Construir mapa de configurações de mão: {mao_id -> url_imagem}.
    3. Organizar as palavras por assunto.
    4. Para cada palavra, baixar vídeo + imagem do sinal + imagem da mão.
    5. Salvar em `data/raw_videos/{assunto}/{palavra}/` com nomes padronizados.
    6. Registrar progresso em `data/download_log.csv` para retomada.

Restrições (.ai/skills/python_data_etl.md):
    - Todas as exceções de rede tratadas explicitamente com logging.
    - Tipagem estática em todas as funções.
"""

import re
import csv
import json
import time
import logging
import requests
from pathlib import Path
from typing import Any

# ──────────────────────────────────────────────────────────────────────────────
# 1. Configuração Geral
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sinais.downloader")

# URLs e Caminhos
BASE_URL_VIDEOS  = "https://dicionario.ines.gov.br/public/media/palavras/videos/"
BASE_URL_IMAGES  = "https://dicionario.ines.gov.br/public/media/palavras/images/"
BASE_URL_HANDS   = "https://dicionario.ines.gov.br/public/media/mao/"
URL_PALAVRAS_JS  = "https://dicionario.ines.gov.br/public/site/js/palavras.js"
URL_ASSUNTOS_JS  = "https://dicionario.ines.gov.br/public/site/js/assuntos.js"
URL_MAO_JS       = "https://dicionario.ines.gov.br/public/site/js/mao.js"

ROOT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw_videos"
LOG_FILE      = Path(__file__).resolve().parent.parent / "data" / "download_log.csv"

# Controle de taxa: espera entre requests (em segundos) para não sobrecarregar o servidor.
DELAY_BETWEEN_REQUESTS: float = 0.3

# HTTP Headers para não ser bloqueado como bot genérico.
HEADERS: dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (SINAIS-LIBRAS-Research/1.0; Educational Use)",
    "Accept": "video/mp4,*/*;q=0.9",
    "Referer": "https://dicionario.ines.gov.br/",
}

# ──────────────────────────────────────────────────────────────────────────────
# 2. Funções de Extração de Dados (ETL Inicial)
# ──────────────────────────────────────────────────────────────────────────────

def _fetch_js_as_json(url: str, var_name: str) -> list[dict[str, Any]]:
    """
    Busca um arquivo .js público do INES e extrai a variável JavaScript
    especificada como JSON Python.

    Args:
        url: URL do arquivo JavaScript a ser buscado.
        var_name: Nome da variável JS a ser extraída (ex: 'palavras', 'assuntos').

    Returns:
        Lista de dicionários com os dados extraídos.

    Raises:
        ValueError: Se o padrão da variável não for encontrado no JS.
        requests.RequestException: Em caso de falha de rede.
    """
    log.info(f"Buscando: {url}")
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()

    # Padrão: var <nome> = [{...}];
    pattern = rf"var\s+{re.escape(var_name)}\s*=\s*(\[.*?\]);"
    match = re.search(pattern, response.text, re.DOTALL)

    if not match:
        raise ValueError(f"Variável '{var_name}' não encontrada no arquivo: {url}")

    return json.loads(match.group(1))


def build_assunto_map(assuntos: list[dict[str, Any]]) -> dict[str, str]:
    """
    Constrói um mapa de {id_assunto: nome_assunto} para lookup rápido.
    """
    return {str(a["id"]): a["nome"] for a in assuntos}


def build_hand_map(maos: list[dict[str, Any]]) -> dict[str, str]:
    """
    Constrói um mapa de {mao_id: url_imagem} a partir do mao.js do INES.

    Cada palavra contém um campo `mao` (ID), que referencia a configuração
    inicial da mão usada no sinal. Este mapa permite resolver o ID para
    o nome do arquivo .jpg correspondente.

    Args:
        maos: Lista de dicts com campos 'id' e 'url' (ex: {'id': '131', 'url': 'cg63.jpg'}).

    Returns:
        Dicionário de mapeamento {mao_id -> nome_arquivo_jpg}.
    """
    return {str(m["id"]): m["url"] for m in maos}


def sanitize_folder_name(name: str) -> str:
    """
    Remove caracteres inválidos para nomes de pasta no Windows e Linux.
    """
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    return cleaned.strip().replace(" ", "_")[:80]


# ──────────────────────────────────────────────────────────────────────────────
# 3. Controle de Log e Progresso (Retomada de Downloads)
# ──────────────────────────────────────────────────────────────────────────────

def load_download_log(log_path: Path) -> set[str]:
    """
    Carrega os identificadores de palavras já baixadas do log CSV.
    Permite retomar downloads interrompidos sem reprocessar o que já existe.
    """
    downloaded_ids: set[str] = set()
    if not log_path.exists():
        return downloaded_ids

    with open(log_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "OK":
                downloaded_ids.add(row["ident"])

    log.info(f"Progresso anterior carregado: {len(downloaded_ids)} entradas já baixadas.")
    return downloaded_ids


def append_download_log(
    log_path: Path,
    ident: str,
    palavra: str,
    video_file: str,
    image_file: str,
    hand_file: str,
    status: str,
    message: str = "",
) -> None:
    """
    Registra o resultado de um download no arquivo de log CSV.
    Agora inclui as colunas de imagem do sinal e configuração de mão.
    """
    file_exists = log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        fieldnames = ["ident", "palavra", "video_file", "image_file", "hand_file", "status", "message"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "ident": ident,
            "palavra": palavra,
            "video_file": video_file,
            "image_file": image_file,
            "hand_file": hand_file,
            "status": status,
            "message": message,
        })


# ──────────────────────────────────────────────────────────────────────────────
# 4. Lógica Principal de Download
# ──────────────────────────────────────────────────────────────────────────────

def download_file(url: str, dest_path: Path) -> bool:
    """
    Baixa qualquer arquivo (vídeo, imagem) via HTTP streaming e salva localmente.

    Args:
        url: URL completa do arquivo a ser baixado.
        dest_path: Caminho completo onde o arquivo deve ser salvo.

    Returns:
        True se o download foi bem-sucedido, False caso contrário.
    """
    try:
        response = requests.get(url, headers=HEADERS, stream=True, timeout=30)
        response.raise_for_status()

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True

    except requests.exceptions.HTTPError as e:
        log.warning(f"HTTP {e.response.status_code} para '{url}'")
        return False
    except requests.exceptions.ConnectionError:
        log.error(f"Erro de conexão para '{url}'. Verifique sua rede.")
        return False
    except requests.exceptions.Timeout:
        log.error(f"Timeout ao baixar '{url}'.")
        return False


def run_downloader(assunto_filter: str | None = None) -> None:
    """
    Função principal do pipeline de download.

    Para cada palavra do assunto selecionado, baixa 3 artefatos:
      - video.mp4        : execução do sinal em LIBRAS
      - sinal.jpg        : foto estática do sinal
      - configuracao_mao.jpg : configuração inicial da mão (lookup em mao.js)

    Estrutura de saída:
      data/raw_videos/{ASSUNTO}/{PALAVRA}/
          ├── video.mp4
          ├── sinal.jpg
          └── configuracao_mao.jpg

    Args:
        assunto_filter: Filtro de assunto (ex: 'SENTIMENTOS'). None = baixa tudo.
    """
    ROOT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 1. Carrega os 3 arquivos JS do INES.
    palavras    = _fetch_js_as_json(URL_PALAVRAS_JS, "palavras")
    assuntos    = _fetch_js_as_json(URL_ASSUNTOS_JS, "assuntos")
    maos        = _fetch_js_as_json(URL_MAO_JS, "mao")
    assunto_map = build_assunto_map(assuntos)
    hand_map    = build_hand_map(maos)

    log.info(f"Vocabulário carregado: {len(palavras)} palavras | {len(assuntos)} assuntos | {len(hand_map)} configurações de mão")

    # 2. Aplica filtro de assunto se especificado.
    if assunto_filter:
        assunto_ids = {
            aid for aid, aname in assunto_map.items()
            if assunto_filter.upper() in aname.upper()
        }
        palavras = [p for p in palavras if str(p.get("assunto", "")) in assunto_ids]
        log.info(f"Filtro '{assunto_filter}' aplicado: {len(palavras)} palavras selecionadas.")

    # 3. Carrega IDs já baixados para retomada automática.
    downloaded_ids = load_download_log(LOG_FILE)

    # 4. Itera e baixa os 3 artefatos por palavra.
    total         = len(palavras)
    success_count = 0
    skip_count    = 0
    error_count   = 0

    for i, palavra in enumerate(palavras, start=1):
        ident      = str(palavra.get("ident", ""))
        nome       = palavra.get("palavra", "DESCONHECIDA")
        video_file = palavra.get("video", "")
        image_file = palavra.get("image", "")
        mao_id     = str(palavra.get("mao", ""))
        assunto_id = str(palavra.get("assunto", "1"))
        hand_file  = hand_map.get(mao_id, "")

        # Pula palavras sem vídeo .mp4.
        if not video_file or not video_file.endswith(".mp4"):
            log.debug(f"[{i}/{total}] SKIP '{nome}' - sem vídeo .mp4 disponível.")
            skip_count += 1
            continue

        # Pula se já baixado com sucesso (retomada).
        if ident in downloaded_ids:
            skip_count += 1
            continue

        # Constrói o diretório de destino organizado por assunto/palavra.
        assunto_name = sanitize_folder_name(assunto_map.get(assunto_id, "SEM_ASSUNTO"))
        palavra_name = sanitize_folder_name(nome)
        word_dir     = ROOT_DATA_DIR / assunto_name / palavra_name

        video_exists = (word_dir / video_file).exists()
        image_exists = (word_dir / "sinal.jpg").exists()
        hand_exists  = (word_dir / "configuracao_mao.jpg").exists()

        # Verifica se todos os artefatos já existem (retomada sem log).
        if video_exists and image_exists and hand_exists:
            append_download_log(LOG_FILE, ident, nome, video_file, image_file, hand_file, "OK", "arquivo_existente")
            skip_count += 1
            continue

        log.info(f"[{i}/{total}] '{nome}' → {assunto_name}/")

        # ── Artefato 1: Vídeo .mp4 ────────────────────────────────────────────
        if not video_exists:
            video_ok = download_file(BASE_URL_VIDEOS + video_file, word_dir / video_file)
            time.sleep(DELAY_BETWEEN_REQUESTS)
        else:
            video_ok = True
            log.debug(f"  Vídeo já existe, pulando: {video_file}")

        # ── Artefato 2: Imagem do sinal .jpg ──────────────────────────────────
        image_ok = image_exists  # Já existia antes de tentar
        if image_file and not image_exists:
            image_ok = download_file(BASE_URL_IMAGES + image_file, word_dir / "sinal.jpg")
            time.sleep(DELAY_BETWEEN_REQUESTS)

        # ── Artefato 3: Configuração de mão .jpg ────────────────────────────
        hand_ok = hand_exists  # Já existia antes de tentar
        if hand_file and not hand_exists:
            hand_ok = download_file(BASE_URL_HANDS + hand_file, word_dir / "configuracao_mao.jpg")
            time.sleep(DELAY_BETWEEN_REQUESTS)

        # Registra resultado no log.
        if video_ok:
            success_count += 1
            status  = "OK"
            message = f"imagem={'OK' if image_ok else 'ERRO'} | mao={'OK' if hand_ok else 'ERRO'}"
        else:
            error_count += 1
            status  = "ERRO"
            message = "video_download_falhou"

        append_download_log(LOG_FILE, ident, nome, video_file, image_file, hand_file, status, message)

    log.info(f"\n{'='*55}")
    log.info(f"✅ Download concluído!")
    log.info(f"   Palavras baixadas (vídeo OK) : {success_count}")
    log.info(f"   Puladas (já existentes)     : {skip_count}")
    log.info(f"   Erros de vídeo              : {error_count}")
    log.info(f"   Log salvo em                : {LOG_FILE}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Ponto de Entrada com CLI Básico
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="S.I.N.A.I.S - Downloader de Vídeos do Dicionário INES (LIBRAS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  # Baixar TUDO (pode demorar muito)
  python scripts/download_ines_videos.py

  # Baixar apenas vídeos do assunto 'FRUTA'
  python scripts/download_ines_videos.py --assunto FRUTA

  # Baixar apenas 'ANIMAL'
  python scripts/download_ines_videos.py --assunto ANIMAL

Assuntos disponíveis no dicionário INES:
  ALIMENTO/BEBIDA, ANIMAL/INSETO/PEIXE/AVE, ANO SIDERAL,
  APARELHO/MÁQUINA, CASA, COR/FORMA, CORPO, ESPORTE/DIVERSÃO,
  FAMÍLIA, FRUTA, HIGIENE/SAÚDE, LEGUME/VERDURA, MATÉRIA/SUBSTÂNCIA,
  NUMERAL/DINHEIRO, PAÍS/ESTADO/CIDADE, PLANTA/FLOR/NATUREZA,
  PROFISSÃO/TRABALHO, SENTIMENTOS, TRANSPORTE/VEÍCULO, VESTUÁRIO/COMPLEMENTOS
        """,
    )
    parser.add_argument(
        "--assunto",
        type=str,
        default=None,
        help="Filtrar por assunto (ex: FRUTA, ANIMAL). Sem filtro = baixa TUDO.",
    )
    args = parser.parse_args()

    run_downloader(assunto_filter=args.assunto)
