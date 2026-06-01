# 📜 scripts/ — Scripts do Pipeline S.I.N.A.I.S

Esta pasta contém os scripts utilitários do projeto, organizados por fase do pipeline.

---

## ✅ Fase 1 — Coleta de Dados (Implementada)

### `inspect_ines_vocabulary.py`
Analisa o vocabulário disponível no Dicionário Digital do INES **sem baixar nada**.
Útil para explorar os assuntos disponíveis e decidir o que coletar.

```bash
python scripts/inspect_ines_vocabulary.py
```

**Saída:** `data/vocabulario_ines.csv`

---

### `download_ines_videos.py`
Baixa os artefatos por palavra do Dicionário INES:
- `{nome}.mp4` — vídeo do sinal em LIBRAS
- `configuracao_mao.jpg` — configuração inicial da mão

```bash
# Baixar tudo (7.000+ palavras — pode demorar horas)
python scripts/download_ines_videos.py

# Baixar apenas um assunto
python scripts/download_ines_videos.py --assunto SENTIMENTOS
```

**Assuntos disponíveis:**
`ALIMENTO/BEBIDA`, `ANIMAL/INSETO/PEIXE/AVE`, `ANO SIDERAL`, `APARELHO/MÁQUINA`,
`CASA`, `COR/FORMA`, `CORPO`, `ESPORTE/DIVERSÃO`, `FAMÍLIA`, `FRUTA`,
`HIGIENE/SAÚDE`, `LEGUME/VERDURA`, `MATÉRIA/SUBSTÂNCIA`, `NUMERAL/DINHEIRO`,
`PAÍS/ESTADO/CIDADE`, `PLANTA/FLOR/NATUREZA`, `PROFISSÃO/TRABALHO`,
`SENTIMENTOS`, `TRANSPORTE/VEÍCULO`, `VESTUÁRIO/COMPLEMENTOS`

**Saída:** `data/raw_videos/{ASSUNTO}/{PALAVRA}/`

**Retomada automática:** Se interrompido, rode o mesmo comando novamente. O script consulta `data/download_log.csv` e pula o que já foi baixado.

---

## ✅ Fase 2 — Processamento de Dados (Implementada)
### `extract_features.py`
Lê os vídeos em .mp4 baixados na Fase 1 e utiliza o MediaPipe para extrair as coordenadas espaciais (x, y, z) dos pontos-chave (landmarks) das mãos, quadro a quadro.
Este script converte a dinâmica temporal do vídeo em matrizes numéricas, preparando os dados brutos para o treinamento da inteligência artificial.

```bash
python scripts/extract_features.py
```

**Entrada:** Vídeos locais na pasta data/raw_videos/

**Saída:** Matrizes temporais salvas em data/processed_features/{nome_do_video}.npy

---

## 🔜 Próximas Fases (A Definir)

| Script | Fase | Status |
|--------|------|--------|
| `data_augmentation.py` | Fase 2b — Aumento de dados | A implementar |
| `train.py` | Fase 3 — Treinamento do classificador | A implementar |
