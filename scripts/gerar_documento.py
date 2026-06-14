"""
Gera o documento conceitual do projeto S.I.N.A.I.S em .docx.
"""
from docx import Document
from docx.shared import Pt, RGBColor, Cm, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os

doc = Document()

# ── Estilos base ────────────────────────────────────────────────────────────
style_normal = doc.styles['Normal']
style_normal.font.name = 'Arial'
style_normal.font.size = Pt(11)

for h_name, size, bold in [('Heading 1', 16, True), ('Heading 2', 13, True), ('Heading 3', 11, True)]:
    s = doc.styles[h_name]
    s.font.name = 'Arial'
    s.font.size = Pt(size)
    s.font.bold = bold
    s.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)

# ── Helpers ─────────────────────────────────────────────────────────────────
def titulo(texto, nivel=1):
    p = doc.add_heading(texto, level=nivel)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    return p

def corpo(texto, negrito_partes=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    if negrito_partes is None:
        run = p.add_run(texto)
        run.font.name = 'Arial'
        run.font.size = Pt(11)
    else:
        # negrito_partes: lista de (texto, negrito)
        for t, nb in negrito_partes:
            run = p.add_run(t)
            run.font.name = 'Arial'
            run.font.size = Pt(11)
            run.bold = nb
    return p

def bullet(texto, negrito=None):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_after = Pt(3)
    if negrito:
        r1 = p.add_run(negrito + ': ')
        r1.bold = True
        r1.font.name = 'Arial'
        r1.font.size = Pt(11)
        r2 = p.add_run(texto)
        r2.font.name = 'Arial'
        r2.font.size = Pt(11)
    else:
        r = p.add_run(texto)
        r.font.name = 'Arial'
        r.font.size = Pt(11)
    return p

def caixa_caderno(caderno, topico):
    """Parágrafo destacado indicando o caderno de referência."""
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.8)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(8)
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:fill'), 'EBF3FB')
    p._p.get_or_add_pPr().append(shd)
    r1 = p.add_run('📓 Caderno de referência: ')
    r1.bold = True
    r1.font.name = 'Arial'
    r1.font.size = Pt(10)
    r1.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    r2 = p.add_run(caderno)
    r2.italic = True
    r2.font.name = 'Arial'
    r2.font.size = Pt(10)
    r2.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    if topico:
        r3 = p.add_run(f'  —  {topico}')
        r3.font.name = 'Arial'
        r3.font.size = Pt(10)
        r3.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    return p

def separador():
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'), '4')
    bottom.set(qn('w:color'), 'BBBBBB')
    pBdr.append(bottom)
    pPr.append(pBdr)
    return p


# ════════════════════════════════════════════════════════════════════════════
# CAPA
# ════════════════════════════════════════════════════════════════════════════
p_titulo = doc.add_paragraph()
p_titulo.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_titulo.paragraph_format.space_before = Pt(60)
p_titulo.paragraph_format.space_after = Pt(8)
r = p_titulo.add_run('S.I.N.A.I.S')
r.font.name = 'Arial'
r.font.size = Pt(28)
r.font.bold = True
r.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)

p_sub = doc.add_paragraph()
p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_sub.paragraph_format.space_after = Pt(4)
r = p_sub.add_run('Sistema Integrado de Notação e Aprendizado de Imagens de Sinais')
r.font.name = 'Arial'
r.font.size = Pt(14)
r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

p_desc = doc.add_paragraph()
p_desc.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_desc.paragraph_format.space_after = Pt(60)
r = p_desc.add_run('Fundamentos Conceituais e Referências Bibliográficas\nBaseado nos cadernos de aula do repositório vision-master')
r.font.name = 'Arial'
r.font.size = Pt(11)
r.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
r.italic = True

doc.add_page_break()

# ════════════════════════════════════════════════════════════════════════════
# INTRODUÇÃO
# ════════════════════════════════════════════════════════════════════════════
titulo('1. Introdução')
corpo(
    'O S.I.N.A.I.S é um sistema de tradução de Libras (Língua Brasileira de Sinais) para texto, '
    'construído sobre técnicas de visão computacional e aprendizado profundo. '
    'Este documento descreve cada conceito utilizado no projeto, indicando o caderno de aula '
    'do repositório vision-master onde esse conceito é apresentado e detalhado.'
)
corpo(
    'O pipeline completo do sistema passa pelas seguintes etapas:'
)
bullet('Captura de vídeo via câmera')
bullet('Detecção de landmarks da mão com MediaPipe')
bullet('Extração e normalização de features temporais')
bullet('Treinamento de uma rede neural LSTM para classificação de sinais')
bullet('Inferência em tempo real com exibição do sinal identificado')

doc.add_page_break()

# ════════════════════════════════════════════════════════════════════════════
# 2. PROCESSAMENTO DE IMAGENS
# ════════════════════════════════════════════════════════════════════════════
titulo('2. Processamento de Imagens e Vídeo')

titulo('2.1 Espaços de Cor e Domínio do Valor', nivel=2)
corpo(
    'Todo frame capturado pela câmera chega no formato BGR (padrão do OpenCV). '
    'Para alimentar o MediaPipe, cada frame é convertido para RGB. '
    'Essa conversão envolve o conceito de domínio do valor — a representação numérica '
    'de cada pixel em diferentes espaços de cor.'
)
caixa_caderno(
    '02.1 — Domínios do Valor e Espaço: Detector de Carros',
    'Introdução a espaços de cor, limiares e representação de pixels'
)
caixa_caderno(
    '02.3 — Domínio do Valor: Aprofundamento em Limiarizações',
    'Operações sobre valores de pixel e thresholding'
)

titulo('2.2 Convolução e Filtros', nivel=2)
corpo(
    'A detecção de landmarks pelo MediaPipe é internamente baseada em redes convolucionais '
    '(CNNs). O conceito de convolução — deslizar um kernel sobre a imagem para extrair '
    'características locais — é a operação fundamental dessas redes.'
)
caixa_caderno(
    '04.1 — Convolução Simples para Criação de Filtros',
    'Operação de convolução, criação e aplicação de kernels'
)
caixa_caderno(
    '04.2 — Convolução para Detecção de Bordas e Canny',
    'Filtros derivativos, detector de bordas Canny'
)

titulo('2.3 Extração de Características (Feature Extraction)', nivel=2)
corpo(
    'O script extract_features.py extrai, para cada frame de vídeo, um vetor de 63 valores '
    'representando as coordenadas (x, y, z) dos 21 landmarks da mão. '
    'Esse processo é conceitualmente equivalente à extração de descritores como HOG '
    '(Histogram of Oriented Gradients), onde a imagem bruta é convertida em uma '
    'representação compacta e informativa.'
)
caixa_caderno(
    '09.1 — Métodos Avançados de Descrição de Características: HoG, SIFT, SURF e ORB',
    'Descritores de características, HOG, SIFT, ORB'
)

separador()

# ════════════════════════════════════════════════════════════════════════════
# 3. MEDIDAS DE DISTÂNCIA E CLASSIFICAÇÃO CLÁSSICA
# ════════════════════════════════════════════════════════════════════════════
titulo('3. Medidas de Distância e Classificação Clássica')
corpo(
    'Antes das redes neurais, abordagens clássicas de classificação usam métricas de '
    'distância para comparar amostras. O kNN (k-Nearest Neighbors) classifica uma nova '
    'amostra pela classe dos k vizinhos mais próximos no espaço de features — '
    'conceitualmente simples e útil como baseline para avaliar se uma representação '
    'de features é discriminativa.'
)
corpo(
    'No contexto do S.I.N.A.I.S, os vetores de landmarks (shape 63) poderiam ser '
    'comparados diretamente por distância euclidiana — uma alternativa antes de usar LSTM.'
)
caixa_caderno(
    '03.1 — Medidas de Distância Básicas e Técnicas Métricas',
    'Distância euclidiana, Manhattan, Minkowski'
)
caixa_caderno(
    '03.2 — kNN: Similaridade como Distância entre Imagens',
    'Classificador k-NN aplicado a imagens'
)

separador()

# ════════════════════════════════════════════════════════════════════════════
# 4. DEEP LEARNING — FUNDAMENTOS
# ════════════════════════════════════════════════════════════════════════════
titulo('4. Aprendizado Profundo — Fundamentos')

titulo('4.1 Introdução às Redes Neurais', nivel=2)
corpo(
    'Uma rede neural artificial é composta por camadas de neurônios que aplicam '
    'transformações lineares seguidas de funções de ativação não-lineares. '
    'O treinamento ocorre por retropropagação (backpropagation): a rede faz uma predição, '
    'calcula o erro pela função de loss, e ajusta os pesos via gradiente descendente.'
)
bullet('Função de ativação', 'GELU')
bullet('Função de loss', 'CrossEntropyLoss — mede distância entre distribuição predita e real')
bullet('Otimizador', 'AdamW — gradiente descendente com momentum e regularização de peso')
bullet('Gradient clipping', 'Limita a magnitude do gradiente para estabilizar o treino')
caixa_caderno(
    '16.1 — Deep Learning: Introdução',
    'Arquitetura de redes neurais, backpropagation, função de loss, otimizadores'
)

titulo('4.2 Classificação de Imagens com Redes Profundas', nivel=2)
corpo(
    'O projeto usa o mesmo paradigma dos cadernos de classificação: um encoder '
    'extrai uma representação vetorial da entrada (aqui, via LSTM em vez de CNN), '
    'e uma cabeça classificadora projeta esse vetor no número de classes. '
    'O processo de treino, validação e teste segue a mesma estrutura.'
)
bullet('Divisão dos dados em treino / validação / teste (80/10/10)')
bullet('Salvamento do melhor modelo pelo val_acc')
bullet('Avaliação final no conjunto de teste isolado')
caixa_caderno(
    '12.2 — Classificação de Imagens com fast.ai v2',
    'Pipeline completo: DataLoader, treino, métricas, avaliação'
)
caixa_caderno(
    '12.3 — Classificação de Imagens com EfficientNet',
    'Transfer learning, fine-tuning, métricas de avaliação'
)

titulo('4.3 Mecanismo de Atenção e Transformers', nivel=2)
corpo(
    'O mecanismo de atenção permite que o modelo pese a importância de cada passo '
    'temporal na sequência ao fazer a predição final. O LSTM bidirecional do S.I.N.A.I.S '
    'é uma arquitetura recorrente; o Transformer seria uma alternativa mais poderosa '
    'para séries temporais longas, pois calcula atenção sobre toda a sequência de uma vez.'
)
caixa_caderno(
    '17.1 — Transformers: Mecanismo de Atenção',
    'Self-attention, multi-head attention, arquitetura Transformer'
)

separador()

# ════════════════════════════════════════════════════════════════════════════
# 5. LSTM E DADOS SEQUENCIAIS
# ════════════════════════════════════════════════════════════════════════════
titulo('5. LSTM — Classificação de Sequências Temporais')

titulo('5.1 O Problema Temporal', nivel=2)
corpo(
    'Um sinal de Libras não é uma imagem estática — é um movimento no tempo. '
    'Cada vídeo gera uma sequência de vetores de landmarks: '
    'tensor de shape (N_frames, 63). '
    'Para classificar corretamente, o modelo precisa considerar todos os frames '
    'e sua ordem temporal, não apenas um instante isolado.'
)

titulo('5.2 Redes Neurais Recorrentes (RNN) e LSTM', nivel=2)
corpo(
    'Uma RNN processa sequências passando um estado oculto de um passo para o '
    'próximo, acumulando contexto temporal. O LSTM (Long Short-Term Memory) '
    'resolve o problema do gradiente que desaparece (vanishing gradient) das '
    'RNNs simples usando três portões: entrada, esquecimento e saída.'
)
bullet('Portão de esquecimento (forget gate)', 'decide o que descartar do estado anterior')
bullet('Portão de entrada (input gate)', 'decide o que adicionar ao estado')
bullet('Portão de saída (output gate)', 'decide o que expor como saída')

titulo('5.3 LSTM Bidirecional', nivel=2)
corpo(
    'O modelo do S.I.N.A.I.S usa um LSTM bidirecional: processa a sequência '
    'de frames tanto da esquerda para direita quanto da direita para esquerda, '
    'dobrando a capacidade de capturar padrões temporais. '
    'A saída do último frame (hidden_dim × 2 = 512) é passada à cabeça classificadora.'
)

titulo('5.4 Padding e Sequências de Tamanho Variável', nivel=2)
corpo(
    'Vídeos diferentes têm números de frames diferentes. Para criar batches homogêneos, '
    'aplica-se padding: sequências curtas recebem frames de zeros no início (pre-padding) '
    'até atingir MAX_SEQ_LEN = 30 frames. Sequências longas são truncadas mantendo '
    'os últimos 30 frames (parte final do sinal).'
)

separador()

# ════════════════════════════════════════════════════════════════════════════
# 6. NORMALIZAÇÃO E DATA AUGMENTATION
# ════════════════════════════════════════════════════════════════════════════
titulo('6. Normalização e Data Augmentation')

titulo('6.1 Normalização de Landmarks', nivel=2)
corpo(
    'Os valores brutos de landmark retornados pelo MediaPipe variam conforme a '
    'posição da mão na tela e a distância da câmera. Para remover essa variação '
    'e tornar o modelo invariante à posição e escala, aplicamos:'
)
bullet('Translação ao pulso', 'subtrai as coordenadas do landmark 0 (pulso) de todos os outros')
bullet('Normalização de escala', 'divide pela distância máxima de qualquer landmark ao pulso')
corpo(
    'Após a normalização, o sinal fica centrado na origem e confinado ao intervalo [-1, 1], '
    'independente de onde a mão estava na tela.'
)
caixa_caderno(
    '09.1 — Métodos Avançados de Descrição: HoG, SIFT, SURF e ORB',
    'Invariância a transformações (escala, rotação, translação) em descritores'
)

titulo('6.2 Data Augmentation', nivel=2)
corpo(
    'Com apenas 1 vídeo por sinal, o dataset é insuficiente para treinar uma rede '
    'generalizável. Data augmentation resolve isso criando variações artificiais do '
    'mesmo sinal a cada epoch — o modelo nunca vê exatamente o mesmo exemplo duas vezes.'
)
bullet('Ruído gaussiano', 'simula imprecisão do detector de landmarks')
bullet('Escala aleatória ±20%', 'simula a mão em diferentes distâncias da câmera')
bullet('Deslocamento espacial XY', 'simula diferentes posições da mão na tela')
bullet('Time-warping ±30%', 'simula o sinal feito em velocidades diferentes')
bullet('Espelhamento horizontal', 'simula a mão espelhada (destro vs. canhoto)')
bullet('Shift temporal ±4 frames', 'simula atraso no início do sinal')
caixa_caderno(
    '12.2 — Classificação de Imagens com fast.ai v2',
    'Augmentation com Albumentations: flip, rotação, escala, brilho'
)
caixa_caderno(
    '12.3 — Classificação de Imagens com EfficientNet',
    'Pipeline de augmentation e normalização com ImageNet stats'
)

separador()

# ════════════════════════════════════════════════════════════════════════════
# 7. SEGMENTAÇÃO SEMÂNTICA (CONTEXTO)
# ════════════════════════════════════════════════════════════════════════════
titulo('7. Detecção e Segmentação — Contexto para o MediaPipe')
corpo(
    'O MediaPipe Hand Landmarker, internamente, realiza duas tarefas sequenciais: '
    'detecção da região da mão na imagem (bounding box) e '
    'regressão dos 21 landmarks dentro dessa região. '
    'Essas tarefas são análogas aos problemas de detecção de objetos e segmentação '
    'estudados nos cadernos da série 14 e 15.'
)
caixa_caderno(
    '14.1 — Detecção de Objetos: YOLOv7',
    'Detecção de objetos em tempo real, bounding boxes, confiança'
)
caixa_caderno(
    '15.1.1 — Segmentação Semântica: U-Net com ResNet',
    'Segmentação pixel a pixel, encoder-decoder, U-Net'
)

separador()

# ════════════════════════════════════════════════════════════════════════════
# 8. INFERÊNCIA EM TEMPO REAL
# ════════════════════════════════════════════════════════════════════════════
titulo('8. Inferência em Tempo Real (testar_camera.py)')

titulo('8.1 Acesso à Câmera e Exibição de Vídeo', nivel=2)
corpo(
    'O script testar_camera.py captura frames da webcam com cv2.VideoCapture(0), '
    'processa cada frame com MediaPipe e exibe o resultado com cv2.imshow(). '
    'Esse padrão de loop de captura e exibição é idêntico ao dos cadernos de webcam.'
)
caixa_caderno(
    'access_webcam_matplotlib_notebook.ipynb',
    'Acesso à câmera, loop de captura, exibição de frames em tempo real'
)
caixa_caderno(
    'Realtime_video_ipython_py3.ipynb',
    'Streaming de vídeo em tempo real no Python'
)

titulo('8.2 Buffer de Frames e Suavização de Predição', nivel=2)
corpo(
    'O modelo precisa de 30 frames para fazer uma predição. '
    'Um deque (fila circular) de tamanho 30 acumula os últimos frames. '
    'Para evitar "flickering" (predição saltando a cada frame), o script mantém '
    'um histórico das últimas 8 predições e exibe a mais frequente — '
    'técnica chamada de suavização por votação majoritária (majority voting).'
)

titulo('8.3 Softmax e Limiar de Confiança', nivel=2)
corpo(
    'Os logits brutos da rede são convertidos em probabilidades pela função softmax. '
    'Só são exibidas predições com confiança acima de um limiar configurável '
    '(padrão: 40%). Isso evita exibir sinais quando nenhuma mão está presente '
    'ou quando o modelo está incerto.'
)
caixa_caderno(
    '16.1 — Deep Learning: Introdução',
    'Função softmax, interpretação de probabilidades, limiar de decisão'
)

separador()

# ════════════════════════════════════════════════════════════════════════════
# 9. MÉTRICAS DE AVALIAÇÃO
# ════════════════════════════════════════════════════════════════════════════
titulo('9. Métricas de Avaliação')
corpo(
    'O projeto monitora as seguintes métricas durante o treinamento:'
)
bullet('train_acc / val_acc / test_acc', 'acurácia: fração de predições corretas')
bullet('train_loss / val_loss / test_loss', 'CrossEntropy Loss: qualidade probabilística do modelo')
bullet('Acaso esperado', '1 / num_classes — baseline de um modelo aleatório')
bullet('Overfitting', 'val_acc muito menor que train_acc indica memorização sem generalização')
bullet('Underfitting', 'ambas as acurácias baixas indicam modelo insuficiente ou dados ruins')
caixa_caderno(
    '12.2 — Classificação de Imagens com fast.ai v2',
    'Métricas: accuracy, F1-score, precision, recall, curva de loss'
)
caixa_caderno(
    '12.3 — Classificação de Imagens com EfficientNet',
    'Avaliação por classe, matriz de confusão, top-k accuracy'
)

separador()

# ════════════════════════════════════════════════════════════════════════════
# 10. TABELA DE REFERÊNCIA RÁPIDA
# ════════════════════════════════════════════════════════════════════════════
titulo('10. Tabela de Referência Rápida')
corpo('Resumo dos conceitos do projeto e seus cadernos de referência no vision-master:')

table = doc.add_table(rows=1, cols=3)
table.style = 'Table Grid'

# Cabeçalho
hdr = table.rows[0].cells
for cell, texto in zip(hdr, ['Conceito', 'Onde é usado no projeto', 'Caderno vision-master']):
    cell.text = texto
    run = cell.paragraphs[0].runs[0]
    run.bold = True
    run.font.name = 'Arial'
    run.font.size = Pt(10)
    cell._tc.get_or_add_tcPr().append(OxmlElement('w:shd'))
    shd = cell._tc.tcPr.find(qn('w:shd'))
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:fill'), 'D5E8F0')

linhas = [
    ('Espaços de cor / BGR→RGB',         'extract_features.py, testar_camera.py',             '02.1 / 02.3'),
    ('Convolução',                        'Operação interna do MediaPipe (CNN)',                '04.1 / 04.2'),
    ('Extração de features (landmarks)',  'extract_features.py',                               '09.1'),
    ('Medidas de distância / kNN',        'Baseline de comparação com LSTM',                   '03.1 / 03.2'),
    ('Redes neurais — fundamentos',       'lstm_classifier.py, train.py',                      '16.1'),
    ('LSTM bidirecional',                 'LIBRASClassifier (lstm_classifier.py)',              '17.1'),
    ('Classificação com deep learning',   'train.py — pipeline treino/val/teste',               '12.2 / 12.3'),
    ('Data augmentation',                 'train.py — função aumentar()',                       '12.2 / 12.3'),
    ('Normalização de features',          'train.py — função normalizar_landmarks()',           '09.1'),
    ('Detecção de objetos',               'MediaPipe: detecta região da mão',                  '14.1'),
    ('Segmentação / regressão de pontos', 'MediaPipe: regressão de 21 landmarks',               '15.1.1'),
    ('Câmera em tempo real',              'testar_camera.py',                                  'access_webcam / Realtime_video'),
    ('Softmax e confiança',               'testar_camera.py — limiar de predição',             '16.1'),
    ('Métricas de avaliação',             'train.py — relatório final',                        '12.2 / 12.3'),
    ('Mecanismo de atenção',              'Alternativa ao LSTM (arquitetura futura)',           '17.1'),
]

for conceito, uso, caderno in linhas:
    row = table.add_row().cells
    for cell, texto in zip(row, [conceito, uso, caderno]):
        cell.text = texto
        cell.paragraphs[0].runs[0].font.name = 'Arial'
        cell.paragraphs[0].runs[0].font.size = Pt(10)

doc.add_paragraph()

# ════════════════════════════════════════════════════════════════════════════
# 11. REFERÊNCIAS
# ════════════════════════════════════════════════════════════════════════════
titulo('11. Referências')
bullet('Repositório vision-master — UFSC / LAPIX')
bullet('INES — Dicionário Digital da Língua Brasileira de Sinais: dicionario.ines.gov.br')
bullet('MediaPipe Hand Landmarker — Google: mediapipe.readthedocs.io')
bullet('PyTorch Documentation: pytorch.org/docs')
bullet('OpenCV Documentation: docs.opencv.org')
bullet('Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.')

# ── Salva ───────────────────────────────────────────────────────────────────
caminho = os.path.join('docs', 'Conceitos_SINAIS.docx')
os.makedirs('docs', exist_ok=True)
doc.save(caminho)
print(f'Documento salvo em: {caminho}')
