# S.I.N.A.I.S — Conceitos da Disciplina e Perguntas de Defesa

> Guia de estudo para a apresentação do projeto. Parte 1 mapeia cada conceito
> usado no S.I.N.A.I.S ao caderno correspondente do `vision-master`; Parte 2
> lista perguntas prováveis do professor com respostas prontas, incluindo os
> números reais do projeto.
>
> Complementa o `Conceitos_SINAIS.md` (atenção: aquele documento é anterior à
> migração para 2 mãos — onde ele diz 63 features, hoje são **126**).

---

# PARTE 1 — Conceitos utilizados

## 1.1 Visão geral do pipeline

```
vídeo (webcam ou INES)
  → OpenCV: captura, BGR→RGB, espelhamento            [cadernos 02.x, webcam]
  → MediaPipe Hands: detecção + regressão de landmarks [cadernos 04.x, 14.x, 16.1]
  → vetor por frame: 126 = 2 mãos × 21 pontos × XYZ
  → pré-processamento: recorte de atividade, padding p/ 30 frames,
    normalização por pulso/escala                      [caderno 09.1 — invariância]
  → LSTM bidirecional → logits → softmax               [cadernos 16.1, 17.1]
  → classificação + rejeição OOD por kNN               [cadernos 03.1, 03.2]
```

Divisão de responsabilidades no código: `models/preprocess.py` (todas as
transformações — fonte única para treino e inferência), `models/lstm_classifier.py`
(rede), `scripts/train.py` (treino), `scripts/avaliar_holdout.py` (avaliação),
`scripts/testar_camera.py` (tempo real).

## 1.2 Processamento de imagem clássico

| Conceito | Onde aparece no projeto | Caderno |
|---|---|---|
| Espaços de cor, domínio do valor | `cv2.cvtColor(BGR→RGB)` antes do MediaPipe | 02.1, 02.3 |
| Operações no domínio do espaço | `cv2.flip` (espelhamento p/ consistência treino/webcam) | 02.1 |
| Convolução e kernels | Base das CNNs internas do MediaPipe | 04.1, 04.2 |
| Captura de vídeo em tempo real | `cv2.VideoCapture`, loop de frames, `imshow` | access_webcam, Realtime_video |

## 1.3 Extração e descrição de características

O projeto **não** classifica pixels: converte cada frame num **descritor
geométrico compacto** (landmarks 3D das mãos). É o mesmo princípio dos
descritores clássicos (HOG/SIFT/ORB — caderno **09.1**): trocar a imagem bruta
por uma representação de baixa dimensão, informativa e **invariante** a
transformações irrelevantes. A invariância aqui é obtida explicitamente na
normalização: translação ao pulso (invariância a posição) e divisão pela
distância máxima ao pulso (invariância a escala/distância da câmera).

O MediaPipe em si é um sistema de dois estágios — **detecção** da mão
(bounding box, como YOLO no caderno **14.1**) seguida de **regressão** dos 21
pontos — ambos implementados com CNNs (caderno **16.1**).

## 1.4 Medidas de distância e kNN

Caderno **03.1/03.2** aplicado duas vezes:

1. **Rejeição OOD (out-of-vocabulary)** no `testar_camera.py`: um banco com as
   features da penúltima camada de todas as amostras de treino é construído; na
   inferência, mede-se a **distância euclidiana ao k-ésimo vizinho** (k=5) da
   entrada atual. Se maior que o limiar (percentil 95 das distâncias do próprio
   treino), a entrada é rejeitada como "fora do vocabulário".
2. **Conceitual**: kNN sobre os descritores seria o baseline clássico antes da
   LSTM; a limitação é que kNN ignora a estrutura temporal do sinal.

## 1.5 Aprendizado profundo

| Conceito | No projeto | Caderno |
|---|---|---|
| Neurônio, camadas, backpropagation | `train.py` (loop explícito PyTorch) | 16.1 |
| Função de loss (CrossEntropy) | `nn.CrossEntropyLoss` | 16.1 |
| Otimizador com momentum (AdamW) + weight decay | `torch.optim.AdamW(lr=3e-4, wd=1e-4)` | 16.1 |
| Learning rate schedule | `CosineAnnealingLR` | 12.2 (fit1cycle é o análogo) |
| Gradient clipping | `clip_grad_norm_(1.0)` | 16.1 |
| Dropout e LayerNorm (regularização) | `lstm_classifier.py` (dropout 0.4) | 16.1 |
| Softmax e limiar de confiança | `testar_camera.py`, `avaliar_holdout.py` | 16.1 |
| Split treino/val/teste estratificado | `_split_estratificado` (80/10/10) | 12.2 |
| Early stopping pela val_loss | `train.py` (paciência 15) | 12.2 |
| Matriz de confusão / análise por classe | relatório por palavra do `avaliar_holdout.py` | 12.2 |
| Transfer learning / fine-tuning | NÃO usado (rede treinada do zero) — citar como alternativa | 12.2, 12.3 |
| Atenção / Transformers | NÃO usado — alternativa futura à LSTM | 17.1 |

## 1.6 Sequências temporais: RNN → LSTM

Um sinal de Libras é **movimento**, não pose estática — a entrada é a sequência
`(30, 126)`. A LSTM resolve o *vanishing gradient* das RNNs simples com três
portões (forget/input/output) que controlam uma célula de memória.
**Bidirecional**: a sequência é lida nos dois sentidos e as saídas concatenadas
(hidden 256 → 512), possível porque classificamos uma janela completa (não é
streaming causal). A predição usa o estado do **último frame** — correto porque
o padding é **pré** (zeros no início, sinal no fim).

## 1.7 Dados: augmentação, hold-out e domain shift

- **1 vídeo por palavra** (INES) não treina rede: a **augmentação offline**
  (9 cópias por vídeo) viabiliza o split, e a **augmentação online** (ruído
  gaussiano, escala ±20%, deslocamento XY, time-warp ±30%, espelhamento com
  troca de mãos, shift temporal) gera uma variação nova a cada época — análogo
  direto do Albumentations no caderno 12.2.
- **Métrica otimista vs honesta**: val/teste derivados de augmentação dos
  mesmos vídeos superestimam o desempenho. A métrica honesta é o **hold-out**:
  gravações do usuário real, de takes que nunca entraram no treino.
- **Domain shift**: modelo treinado com 1 sinalizante (INES) não generaliza
  para outro corpo/câmera/iluminação. Resultado experimental do projeto
  (10 palavras mais frequentes): só INES → **45%** no hold-out; INES + 4
  repetições próprias por palavra → **95–100%**.

## 1.8 Decisões de projeto defensáveis

1. **Recorte da janela de atividade** (`recortar_atividade`): mantém apenas os
   frames onde há mão detectada (±2 de margem) antes do padding. Sem isso, os
   vídeos de webcam (sinal no meio do take) viravam tensores 100% zerados —
   bug real encontrado e corrigido no projeto.
2. **Fonte única de pré-processamento**: treino e inferência importam as mesmas
   funções (`models/preprocess.py`). Qualquer divergência entre os dois quebra
   o modelo silenciosamente.
3. **Unificação de variantes na classificação**: o INES tem execuções
   alternativas (QUE1/QUE2). O modelo é treinado com as duas classes, mas na
   classificação as probabilidades são **somadas** e exibidas como "QUE" —
   o usuário não se importa qual variante executou. Sem retreinar nada.
4. **Convenção de mãos**: slot fixo por handedness (`Left`→0:63,
   `Right`→63:126, ausente = zeros) e espelhamento consistente entre extração
   e webcam.

---

# PARTE 2 — Perguntas prováveis e respostas

## Arquitetura e modelo

**P1. Por que vocês usam landmarks do MediaPipe em vez de treinar uma CNN
direto nos pixels do vídeo?**
R: Três razões. (1) *Dados*: uma CNN 3D end-to-end precisa de milhares de
vídeos por classe; nós temos 1 vídeo INES + poucas repetições próprias — os
landmarks são um descritor de 126 dimensões que já elimina fundo, iluminação e
aparência, sobrando só a geometria do gesto. (2) *Custo*: a LSTM sobre
landmarks treina em minutos numa CPU. (3) *Invariância*: sobre landmarks
conseguimos normalizar posição e escala analiticamente, em vez de esperar que a
rede aprenda essas invariâncias. É a mesma filosofia dos descritores clássicos
(HOG/SIFT): representação compacta e invariante antes do classificador.

**P2. Por que LSTM e não uma rede comum (MLP) ou uma CNN?**
R: O sinal é uma **sequência temporal** — a ordem dos frames importa (ex.:
mão subindo vs descendo têm os mesmos frames em ordem inversa). Um MLP sobre a
concatenação dos frames ignoraria o alinhamento temporal e explodiria em
parâmetros. A LSTM processa frame a frame mantendo um estado de memória, e os
portões (forget/input/output) resolvem o vanishing gradient que impedia RNNs
simples de aprender dependências longas.

**P3. O que significa "bidirecional" e por que podem usar isso em tempo real?**
R: Duas LSTMs, uma lendo do frame 1 ao 30 e outra do 30 ao 1; as saídas são
concatenadas (256+256=512). Podemos usar porque a inferência é sobre uma
**janela completa** de 30 frames já capturados (buffer deslizante), não frame a
frame causal — quando a janela é classificada, o "futuro" dela já existe.

**P4. Por que pegar a saída do último frame e não a média de todos?**
R: Com **pre-padding** (zeros no início, sinal no fim), o estado no último
frame já acumulou a sequência real inteira. A média diluiria o estado com os
frames de padding. Testamos essa configuração e é a convenção documentada no
modelo.

**P5. Quantos parâmetros/camadas tem o modelo? Descreva a arquitetura.**
R: Entrada (batch, 30, 126) → LSTM bidirecional, 2 camadas, hidden 256,
dropout 0.4 → estado do último frame (512) → LayerNorm → Dropout →
Linear 512→256 → GELU → Dropout → Linear 256→N classes. ~2,5 M parâmetros
(checkpoint de ~10 MB em float32). O gargalo de 256 evita explosão de
parâmetros quando o vocabulário cresce.

**P6. Um Transformer não seria melhor?**
R: Provavelmente sim para sequências longas — self-attention pesa todos os
frames de uma vez e não sofre com memória sequencial (caderno 17.1). Não usamos
porque com 30 frames e poucos dados a LSTM é suficiente e mais estável de
treinar do zero; Transformer é trabalho futuro natural, junto de
transfer learning de modelos de linguagem de sinais.

## Dados e pré-processamento

**P7. Como uma imagem vira entrada da rede?**
R: O frame chega em BGR (OpenCV), convertemos para RGB e o MediaPipe detecta
até 2 mãos, devolvendo 21 landmarks (x, y, z normalizados) por mão. Montamos um
vetor de 126: mão "Left" nas posições 0–62, "Right" nas 63–125, ausente = zeros.
O vídeo inteiro vira uma matriz (N frames, 126) salva em `.npy`.

**P8. Para que serve a normalização por pulso?**
R: Invariância. Subtrair o pulso de todos os pontos remove a **posição** da mão
na tela; dividir pela distância máxima ao pulso remove a **escala** (perto/longe
da câmera). Sobra apenas a *forma e configuração* da mão, que é o que define o
sinal. Sem isso, o modelo aprenderia "QUE é quando a mão está no canto direito".

**P9. Os vídeos têm comprimentos diferentes. Como viram batches?**
R: Primeiro recortamos a janela de atividade (frames onde há mão, ±2 de
margem); depois ajustamos para 30 frames: sequência longa mantém os últimos 30,
curta recebe zeros no **início** (pre-padding). O recorte é essencial: nos
vídeos de webcam o sinal fica no meio do take, e sem recorte os "últimos 30
frames" podiam ser só braço abaixado — descobrimos isso quando 13 dos 20 takes
de teste viravam tensores zerados.

**P10. Expliquem a augmentação de dados. Por que offline E online?**
R: *Offline*: 9 cópias transformadas de cada `.npy` — sem elas não haveria
amostras suficientes para dividir treino/val/teste com 1 vídeo por palavra.
*Online*: a cada época, cada amostra recebe transformações aleatórias novas
(ruído σ=0.02 simulando imprecisão do MediaPipe, escala ±20%, deslocamento XY,
reamostragem temporal ±30% simulando velocidade, espelhamento — que troca os
slots das mãos, como um espelho real —, shift de ±4 frames). O modelo nunca vê
o mesmo tensor duas vezes, o que combate memorização.

**P11. O que é domain shift e como apareceu no projeto?**
R: É quando a distribuição dos dados de uso difere da de treino. O INES tem
**um** sinalizante profissional; treinando só com ele, o modelo acerta os
próprios vídeos mas falha com outra pessoa (outro corpo, câmera, iluminação,
velocidade). Medimos: 45% no hold-out do usuário vs 95–100% quando 4 repetições
do próprio usuário por palavra entram no treino. Foi a descoberta mais
importante do projeto: **dados do domínio-alvo valem mais que qualquer ajuste
de arquitetura.**

## Treinamento e avaliação

**P12. Como foi dividido o dataset? Por que estratificado?**
R: 80/10/10 por classe (estratificado) — garante que toda palavra apareça nos
três conjuntos; com poucas amostras por classe, um split aleatório simples
poderia deixar classes sem representação na validação. Semente fixa para
reprodutibilidade.

**P13. O que é early stopping e por que monitorar a val_loss e não a val_acc?**
R: Paramos quando a val_loss não melhora por 15 épocas e restauramos o melhor
checkpoint — evita overfitting por excesso de épocas. A loss é preferível à
acurácia porque é contínua: detecta degradação da *confiança* das predições
antes de a acurácia (discreta, com poucas amostras) mudar.

**P14. Como vocês sabem que o modelo não está só decorando?**
R: Três níveis. (1) val/teste separados durante o treino; (2) a régua do acaso:
com N classes, um modelo aleatório acerta 1/N e a loss inicial é ln(N) — se a
loss não fura ln(N) nas primeiras épocas, o modelo não está aprendendo (usamos
isso como diagnóstico: com 500 classes a loss travou em ln(500)=6,21 e o treino
colapsou); (3) o **hold-out**: takes gravados pelo usuário que nunca entraram
no treino nem como augmentação — nossa métrica final é medida neles.

**P15. Qual a acurácia final?**
R: No vocabulário das 10 palavras mais frequentes do PT-BR: **100% (20/20)**
no hold-out com o modelo treinado com INES + vídeos próprios e variantes
QUE1/QUE2 unificadas na classificação (95% num segundo treino — a variação é
ruído de inicialização). O modelo só-INES fica em 45% no mesmo teste, o que
quantifica o domain shift. Escala: com 1 vídeo/palavra + augmentação o treino
funciona até ~300 classes (21% test, 64× o acaso) e colapsa em 500.

**P16. Por que a acurácia de teste durante o treino não é confiável?**
R: Porque o teste interno é formado por augmentações **dos mesmos vídeos** de
treino (com 1 vídeo por palavra não há alternativa) — mede robustez às
transformações, não generalização a novos sinalizantes. Por isso criamos o
hold-out gravado separadamente.

## Inferência em tempo real

**P17. Como funciona o reconhecimento pela webcam?**
R: Um `deque` guarda os últimos 30 vetores de landmarks. A cada frame: gate de
presença de mão (≥30% dos frames com mão), recorte+normalização idênticos ao
treino, forward na LSTM, softmax. Anti-flicker: só entram no histórico
predições acima do limiar de confiança e exibimos a mais frequente das últimas
8 (majority voting). A palavra exibida vem com o vídeo de referência do INES.

**P18. O que acontece se eu fizer um sinal que o modelo não conhece?**
R: Detector de **OOD por kNN**: guardamos as features da penúltima camada de
todas as amostras de treino; se a distância da entrada atual ao 5º vizinho mais
próximo excede o percentil 95 das distâncias internas do treino, exibimos "fora
do vocabulário" em vez de chutar. Limitação honesta: sinais *parecidos* com o
vocabulário (near-OOD) ainda enganam o detector — é limite fundamental do
método, não bug.

**P19. Por que QUE1 e QUE2 aparecem como "QUE" na tela?**
R: São duas execuções alternativas do mesmo sinal no dicionário. Mantemos as
duas classes no treino (mais dados, distribuições distintas), mas na
classificação **somamos as probabilidades** e exibimos a palavra-base — se
QUE1 dá 40% e QUE2 dá 35%, o resultado é QUE com 75%. Melhor que renomear o
argmax (os votos das variantes se somam) e não exige retreino.

**P20. Por que o frame é espelhado na extração?**
R: Consistência de convenção. A webcam exibe imagem espelhada (natural para o
usuário); o MediaPipe rotula "Left/Right" sobre o que vê. Espelhamos os vídeos
do INES na extração para que o rótulo de mão caia no mesmo slot do vetor nos
dois mundos — senão a mão direita do treino seria a esquerda da inferência.

## Perguntas capciosas

**P21. Se eu re-treinar com os vídeos do hold-out, a acurácia sobe. Por que
vocês não fizeram isso?**
R: Porque perderíamos a única medida honesta de generalização. Hold-out é
sagrado: se entra no treino, o número que sai não prevê mais o desempenho com
dados novos.

**P22. 20 amostras de teste não é pouco?**
R: É — cada take vale 5 pontos percentuais, então 95% vs 90% não é diferença
significativa. Tratamos como evidência de ordem de grandeza (95–100% vs 45% é
conclusivo; 95% vs 90% não). O plano de coleta prevê expandir palavras e takes.

**P23. A acurácia de 100% não indica overfitting?**
R: Overfitting seria acertar o treino e errar dados novos — e o hold-out É dado
novo (takes gravados em momento diferente, nunca vistos). Com 9 rótulos o acaso
é 11%; 20/20 tem probabilidade ~10⁻¹⁹ por sorte. O que esse número **não**
garante é generalização para *outras pessoas* — para isso precisaríamos de
hold-out de sinalizantes diferentes do usuário que treinou.

**P24. Por que não usar transfer learning, como nos cadernos 12.x?**
R: Transfer learning pressupõe um backbone pré-treinado no mesmo tipo de
entrada. Para imagens há ImageNet; para sequências de landmarks de Libras não
há um "ImageNet de sinais" disponível. O análogo que usamos é o próprio
MediaPipe — uma rede pré-treinada pelo Google que transferimos como extrator de
features congelado.

**P25. O sistema traduz frases?**
R: Não — classifica sinais isolados de um vocabulário fechado. Tradução
contínua exigiria segmentação temporal de sinais em sequência, tratamento de
transições e gramática da Libras (que não é português sinalizado) — escopo de
pesquisa, não de um semestre.

---

## Ficha técnica (decorar antes da defesa)

| Item | Valor |
|---|---|
| Features por frame | 126 (2 mãos × 21 landmarks × XYZ) |
| Janela temporal | 30 frames, pre-padding, recorte de atividade ±2 |
| Modelo | LSTM bidir, 2 camadas, hidden 256, dropout 0.4, ~2,5 M parâmetros |
| Otimização | AdamW lr 3e-4, wd 1e-4, CosineAnnealing, clip 1.0, batch 16 |
| Early stopping | paciência 15 (val_loss) |
| Dados INES | 5.468 palavras com vídeo (1 por palavra) + 9 augmentações offline |
| Dados próprios (top-10) | 4 reps treino + 2 hold-out por palavra |
| Resultado-chave | hold-out: 45% (só INES) → 95–100% (INES + próprios, QUE unificado) |
| Limite de escala | treina até ~300 classes; colapsa em 500 (loss presa em ln 500) |
| OOD | kNN (k=5) na penúltima camada, limiar = percentil 95 |
