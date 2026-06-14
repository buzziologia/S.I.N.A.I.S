# S.I.N.A.I.S — Fundamentos Conceituais

**Sistema Integrado de Notação e Aprendizado de Imagens de Sinais**  
Baseado nos cadernos de aula do repositório `vision-master`

---

## Sumário

1. [Introdução](#1-introdução)
2. [Processamento de Imagens e Vídeo](#2-processamento-de-imagens-e-vídeo)
3. [Medidas de Distância e Classificação Clássica](#3-medidas-de-distância-e-classificação-clássica)
4. [Aprendizado Profundo — Fundamentos](#4-aprendizado-profundo--fundamentos)
5. [LSTM — Classificação de Sequências Temporais](#5-lstm--classificação-de-sequências-temporais)
6. [Normalização e Data Augmentation](#6-normalização-e-data-augmentation)
7. [Detecção e Segmentação — Contexto para o MediaPipe](#7-detecção-e-segmentação--contexto-para-o-mediapipe)
8. [Inferência em Tempo Real](#8-inferência-em-tempo-real)
9. [Métricas de Avaliação](#9-métricas-de-avaliação)
10. [Tabela de Referência Rápida](#10-tabela-de-referência-rápida)
11. [Referências](#11-referências)

---

## 1. Introdução

O S.I.N.A.I.S é um sistema de tradução de Libras (Língua Brasileira de Sinais) para texto, construído sobre técnicas de visão computacional e aprendizado profundo. Este documento descreve cada conceito utilizado no projeto, indicando o caderno de aula do repositório `vision-master` onde esse conceito é apresentado e detalhado.

O pipeline completo do sistema passa pelas seguintes etapas:

1. Captura de vídeo via câmera
2. Detecção de landmarks da mão com MediaPipe
3. Extração e normalização de features temporais
4. Treinamento de uma rede neural LSTM para classificação de sinais
5. Inferência em tempo real com exibição do sinal identificado

---

## 2. Processamento de Imagens e Vídeo

### 2.1 Espaços de Cor e Domínio do Valor

Todo frame capturado pela câmera chega no formato **BGR** (padrão do OpenCV). Para alimentar o MediaPipe, cada frame é convertido para **RGB** via `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`. Essa conversão envolve o conceito de **domínio do valor** — a representação numérica de cada pixel em diferentes espaços de cor.

> 📓 **Caderno de referência:**  
> `02.1 — Domínios do Valor e Espaço: Detector de Carros` — introdução a espaços de cor, limiares e representação de pixels  
> `02.3 — Domínio do Valor: Aprofundamento em Limiarizações` — operações sobre valores de pixel e thresholding

---

### 2.2 Convolução e Filtros

A detecção de landmarks pelo MediaPipe é internamente baseada em **redes convolucionais (CNNs)**. O conceito de convolução — deslizar um kernel sobre a imagem para extrair características locais — é a operação fundamental dessas redes.

> 📓 **Caderno de referência:**  
> `04.1 — Convolução Simples para Criação de Filtros` — operação de convolução, criação e aplicação de kernels  
> `04.2 — Convolução para Detecção de Bordas e Canny` — filtros derivativos, detector de bordas Canny

---

### 2.3 Extração de Características (Feature Extraction)

O script `extract_features.py` extrai, para cada frame de vídeo, um vetor de **63 valores** representando as coordenadas (x, y, z) dos 21 landmarks da mão. Esse processo é conceitualmente equivalente à extração de descritores como **HOG** (Histogram of Oriented Gradients), onde a imagem bruta é convertida em uma representação compacta e informativa.

> 📓 **Caderno de referência:**  
> `09.1 — Métodos Avançados de Descrição de Características: HoG, SIFT, SURF e ORB` — descritores de features, HOG, SIFT, ORB e invariância a transformações

---

## 3. Medidas de Distância e Classificação Clássica

Antes das redes neurais, abordagens clássicas de classificação usam métricas de distância para comparar amostras. O **kNN (k-Nearest Neighbors)** classifica uma nova amostra pela classe dos k vizinhos mais próximos no espaço de features — conceitualmente simples e útil como baseline.

No contexto do S.I.N.A.I.S, os vetores de landmarks (shape `63`) poderiam ser comparados diretamente por distância euclidiana — uma alternativa antes de usar LSTM.

> 📓 **Caderno de referência:**  
> `03.1 — Medidas de Distância Básicas e Técnicas Métricas` — distância euclidiana, Manhattan, Minkowski  
> `03.2 — kNN: Similaridade como Distância entre Imagens` — classificador k-NN aplicado a imagens

---

## 4. Aprendizado Profundo — Fundamentos

### 4.1 Introdução às Redes Neurais

Uma rede neural artificial é composta por camadas de neurônios que aplicam transformações lineares seguidas de funções de ativação não-lineares. O treinamento ocorre por **retropropagação (backpropagation)**: a rede faz uma predição, calcula o erro pela função de loss, e ajusta os pesos via gradiente descendente.

Componentes usados no S.I.N.A.I.S:

| Componente | Descrição |
|---|---|
| **GELU** | Função de ativação suave, melhor que ReLU para modelos profundos |
| **CrossEntropyLoss** | Mede a distância entre a distribuição predita e a real |
| **AdamW** | Gradiente descendente com momentum e regularização de peso |
| **Gradient clipping** | Limita a magnitude do gradiente para estabilizar o treino |

> 📓 **Caderno de referência:**  
> `16.1 — Deep Learning: Introdução` — arquitetura de redes neurais, backpropagation, função de loss, otimizadores

---

### 4.2 Classificação com Redes Profundas

O projeto usa o mesmo paradigma dos cadernos de classificação: um **encoder** extrai uma representação vetorial da entrada (aqui, via LSTM em vez de CNN), e uma **cabeça classificadora** projeta esse vetor no número de classes.

- Divisão dos dados: **80% treino / 10% validação / 10% teste** (estratificado)
- Salvamento do melhor modelo pelo `val_acc`
- Avaliação final no conjunto de teste isolado

> 📓 **Caderno de referência:**  
> `12.2 — Classificação de Imagens com fast.ai v2` — pipeline completo: DataLoader, treino, métricas, avaliação  
> `12.3 — Classificação de Imagens com EfficientNet` — transfer learning, fine-tuning, métricas de avaliação

---

### 4.3 Mecanismo de Atenção e Transformers

O **mecanismo de atenção** permite que o modelo pese a importância de cada passo temporal ao fazer a predição final. O LSTM bidirecional do S.I.N.A.I.S é uma arquitetura recorrente; o **Transformer** seria uma alternativa mais poderosa para sequências longas, pois calcula atenção sobre toda a sequência de uma vez.

> 📓 **Caderno de referência:**  
> `17.1 — Transformers: Mecanismo de Atenção` — self-attention, multi-head attention, arquitetura Transformer

---

## 5. LSTM — Classificação de Sequências Temporais

### 5.1 O Problema Temporal

Um sinal de Libras não é uma imagem estática — é um **movimento no tempo**. Cada vídeo gera uma sequência de vetores de landmarks: tensor de shape `(N_frames, 63)`. Para classificar corretamente, o modelo precisa considerar todos os frames e sua ordem temporal.

---

### 5.2 RNN e LSTM

Uma **RNN** processa sequências passando um estado oculto de um passo para o próximo. O **LSTM (Long Short-Term Memory)** resolve o problema do gradiente que desaparece (vanishing gradient) usando três portões:

- **Forget gate** — decide o que descartar do estado anterior
- **Input gate** — decide o que adicionar ao estado
- **Output gate** — decide o que expor como saída

---

### 5.3 LSTM Bidirecional

O modelo usa um **LSTM bidirecional**: processa a sequência de frames tanto da esquerda para direita quanto da direita para esquerda, dobrando a capacidade de capturar padrões temporais.

```
Entrada:  (batch, 30, 63)
LSTM:     hidden_dim=256, bidirectional=True  →  saída (batch, 30, 512)
Último frame: (batch, 512)
Classifier: Linear(512→256) → GELU → Linear(256→num_classes)
Saída:    (batch, num_classes)
```

---

### 5.4 Padding de Sequências

Vídeos diferentes têm números de frames diferentes. Para criar batches homogêneos:

- Sequências **curtas** → zeros no início (**pre-padding**) até `MAX_SEQ_LEN = 30`
- Sequências **longas** → mantém os últimos 30 frames (parte final do sinal)

O pre-padding é preferível ao post-padding porque o LSTM lê o sinal real por último, e o estado final reflete o conteúdo verdadeiro.

---

## 6. Normalização e Data Augmentation

### 6.1 Normalização de Landmarks

Os valores brutos de landmark variam conforme a posição da mão na tela e a distância da câmera. Para tornar o modelo invariante, aplicamos por frame:

1. **Translação ao pulso** — subtrai as coordenadas do landmark 0 (pulso) de todos os outros
2. **Normalização de escala** — divide pela distância máxima de qualquer landmark ao pulso

Após a normalização, o sinal fica centrado na origem e confinado a `[-1, 1]`, independente de onde a mão estava na tela.

> 📓 **Caderno de referência:**  
> `09.1 — Métodos Avançados de Descrição: HoG, SIFT, SURF e ORB` — invariância a transformações (escala, rotação, translação) em descritores

---

### 6.2 Data Augmentation

Com apenas **1 vídeo por sinal**, o dataset é insuficiente para treinar uma rede generalizável. Data augmentation cria variações artificiais a cada epoch — o modelo nunca vê exatamente o mesmo exemplo duas vezes.

| Técnica | Simula |
|---|---|
| Ruído gaussiano (`σ=0.02`) | Imprecisão do detector de landmarks |
| Escala aleatória `±20%` | Mão em diferentes distâncias da câmera |
| Deslocamento espacial XY | Diferentes posições da mão na tela |
| Time-warping `±30%` | Sinal feito em velocidades diferentes |
| Espelhamento horizontal | Destro vs. canhoto |
| Shift temporal `±4 frames` | Atraso no início do sinal |

> 📓 **Caderno de referência:**  
> `12.2 — Classificação de Imagens com fast.ai v2` — augmentation com Albumentations: flip, rotação, escala, brilho  
> `12.3 — Classificação de Imagens com EfficientNet` — pipeline de augmentation e normalização

---

## 7. Detecção e Segmentação — Contexto para o MediaPipe

O **MediaPipe Hand Landmarker** realiza internamente duas tarefas sequenciais:

1. **Detecção** da região da mão na imagem (bounding box)
2. **Regressão** dos 21 landmarks dentro dessa região

Essas tarefas são análogas aos problemas de detecção de objetos e segmentação estudados nos cadernos da série 14 e 15.

> 📓 **Caderno de referência:**  
> `14.1 — Detecção de Objetos: YOLOv7` — detecção em tempo real, bounding boxes, confiança  
> `15.1.1 — Segmentação Semântica: U-Net com ResNet` — segmentação pixel a pixel, encoder-decoder

---

## 8. Inferência em Tempo Real

### 8.1 Acesso à Câmera

O script `testar_camera.py` captura frames com `cv2.VideoCapture(0)`, processa com MediaPipe e exibe com `cv2.imshow()`. Esse padrão de loop de captura e exibição é idêntico ao dos cadernos de webcam.

> 📓 **Caderno de referência:**  
> `access_webcam_matplotlib_notebook.ipynb` — acesso à câmera, loop de captura, exibição em tempo real  
> `Realtime_video_ipython_py3.ipynb` — streaming de vídeo em tempo real

---

### 8.2 Buffer de Frames e Suavização

O modelo precisa de 30 frames para fazer uma predição. Um `deque` de tamanho 30 acumula os últimos frames. Para evitar **flickering** (predição saltando a cada frame), o script mantém um histórico das últimas 8 predições e exibe a mais frequente — técnica chamada **majority voting**.

---

### 8.3 Softmax e Limiar de Confiança

Os logits brutos da rede são convertidos em probabilidades pela função **softmax**. Só são exibidas predições com confiança acima de um limiar configurável (padrão: `40%`). Isso evita exibir sinais quando nenhuma mão está presente ou quando o modelo está incerto.

> 📓 **Caderno de referência:**  
> `16.1 — Deep Learning: Introdução` — softmax, interpretação de probabilidades, limiar de decisão

---

## 9. Métricas de Avaliação

| Métrica | Descrição |
|---|---|
| `train_acc / val_acc / test_acc` | Fração de predições corretas |
| `train_loss / val_loss / test_loss` | CrossEntropy Loss: qualidade probabilística |
| **Acaso esperado** | `1 / num_classes` — baseline de um modelo aleatório |
| **Overfitting** | `val_acc` muito menor que `train_acc` → memorização sem generalização |
| **Underfitting** | Ambas as acurácias baixas → modelo insuficiente ou dados ruins |

> 📓 **Caderno de referência:**  
> `12.2 — Classificação de Imagens com fast.ai v2` — acurácia, F1-score, precision, recall, curva de loss  
> `12.3 — Classificação de Imagens com EfficientNet` — avaliação por classe, matriz de confusão

---

## 10. Tabela de Referência Rápida

| Conceito | Onde é usado no projeto | Caderno vision-master |
|---|---|---|
| Espaços de cor / BGR→RGB | `extract_features.py`, `testar_camera.py` | `02.1` / `02.3` |
| Convolução | Operação interna do MediaPipe (CNN) | `04.1` / `04.2` |
| Extração de features (landmarks) | `extract_features.py` | `09.1` |
| Medidas de distância / kNN | Baseline de comparação com LSTM | `03.1` / `03.2` |
| Redes neurais — fundamentos | `lstm_classifier.py`, `train.py` | `16.1` |
| LSTM bidirecional | `LIBRASClassifier` em `lstm_classifier.py` | `17.1` |
| Classificação com deep learning | `train.py` — pipeline treino/val/teste | `12.2` / `12.3` |
| Data augmentation | `train.py` — função `aumentar()` | `12.2` / `12.3` |
| Normalização de features | `train.py` — função `normalizar_landmarks()` | `09.1` |
| Detecção de objetos | MediaPipe: detecta região da mão | `14.1` |
| Segmentação / regressão de pontos | MediaPipe: regressão de 21 landmarks | `15.1.1` |
| Câmera em tempo real | `testar_camera.py` | `access_webcam` / `Realtime_video` |
| Softmax e confiança | `testar_camera.py` — limiar de predição | `16.1` |
| Métricas de avaliação | `train.py` — relatório final | `12.2` / `12.3` |
| Mecanismo de atenção | Alternativa ao LSTM (arquitetura futura) | `17.1` |

---

## 11. Referências

- Repositório **vision-master** — UFSC / LAPIX
- **INES** — Dicionário Digital da Língua Brasileira de Sinais: [dicionario.ines.gov.br](https://dicionario.ines.gov.br)
- **MediaPipe** Hand Landmarker — Google: [mediapipe.readthedocs.io](https://mediapipe.readthedocs.io)
- **PyTorch** Documentation: [pytorch.org/docs](https://pytorch.org/docs)
- **OpenCV** Documentation: [docs.opencv.org](https://docs.opencv.org)
- Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735–1780.
