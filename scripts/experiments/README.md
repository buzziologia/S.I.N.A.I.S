# 🧪 experiments/ - Experimentos (registro de hipóteses testadas)

Cada script aqui testa uma ideia para melhorar o reconhecimento. O objetivo é
**registrar o que já foi testado e o resultado**, pra não re-investigar.

Rodar (via `just`):
```bash
just exp-grupos     # ou: python scripts/experiments/experimento_grupos.py
just exp-config     # ou: python scripts/experiments/experimento_config_mao.py
```

---

## `experimento_grupos.py` - N modelos por grupo + OOD para escolher o modelo
**Hipótese:** dividir as palavras em grupos, treinar 1 modelo por grupo, e na
inferência rodar a entrada em todos e escolher pelo score de OOD (qual modelo a
considera "in-distribution"). Testa 4 scores: **MSP, Energy, KNN, Mahalanobis**.

**Resultado (FRUTA × COR_FORMA, 2 grupos):**
| | comb. acc | erro venceu | P(certo|1) |
|---|---:|---:|---:|
| Oráculo (sabe o grupo) | ~68% | - | - |
| MSP | ~44% | 42% | 58% |
| Energy | ~49% | 36% | 64% |
| **KNN / Mahalanobis** | **~54%** | 29% | 71% |

**Conclusão: ❌ não funciona.** Os experts treinam bem, mas a **seleção** é o
gargalo (problema de conjunto aberto / near-OOD - sinais de outro grupo são mãos
válidas, alta densidade em todo modelo). Mesmo o melhor score (KNN ≈ Maha) erra a
escolha em ~29% **com só 1 competidor**; extrapolando p/ ~156 grupos
(`0.71^155 ≈ 10⁻²³`), a seleção colapsa. Confirma a "rede hierárquica" como beco
sem saída.

---

## `experimento_config_mao.py` - configuração de mão como tarefa auxiliar
**Hipótese:** usar a `configuracao_mao.jpg` (informação privilegiada, só no treino)
como tarefa auxiliar (multi-task), regredindo o vetor 126-d da configuração canônica
junto com a classificação da palavra. Cabeça auxiliar é ignorada na inferência.

**Resultado (top-100, 40 épocas, λ=0.3):**
| | melhor val_acc | test_acc |
|---|---:|---:|
| Baseline | 62.6% | 56.6% |
| Multi-task | 64.6% | **57.6%** |
| **Δ** | +2.0 pt | **+1.0 pt** |

**Conclusão: ❌ ganho desprezível (dentro do ruído).** A configuração de mão é
**constante por classe** → só regulariza, não adiciona informação discriminativa.
Com 1 vídeo/palavra o gargalo é **dado** (mais vídeos reais), não feature.

---

## 📌 Conclusão geral dos experimentos
O gargalo do projeto **não é arquitetura nem features extras** - é **dados**:
1 vídeo real por palavra não generaliza para um sinalizante novo (domain shift).
O caminho de maior impacto é **coletar mais vídeos reais** (`just coletar`) e
**reduzir o vocabulário** às palavras mais usadas (`just train --top_k N`).
