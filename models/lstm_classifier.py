import torch
import torch.nn as nn


class LIBRASClassifier(nn.Module):
    """
    LSTM bidirecional para classificação de sinais em LIBRAS.

    Entrada : (batch, seq, 126)  - sequência temporal de landmarks das 2 mãos
    Saída   : (batch, num_classes) - logits por classe

    Arquitetura: LSTM bidirecional (2 camadas, hidden=256) → estado do último
    frame temporal (correto com pre-padding: zeros no início, sinal no fim) →
    classificador com gargalo intermediário de 256 unidades.
    """

    def __init__(self, input_dim: int = 126, hidden_dim: int = 256,
                 num_layers: int = 2, num_classes: int = 5452,
                 dropout: float = 0.4):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_dim * 2  # bidirecional → 512

        # Gargalo intermediário evita explosão de parâmetros com muitas classes
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, features)
        out, _ = self.lstm(x)
        # Último frame - correto com pre-padding (zeros no início, sinal no fim)
        last = out[:, -1, :]
        return self.classifier(last)


def carregar_modelo(caminho_pesos: str, num_classes: int, device: str = "cpu") -> LIBRASClassifier:
    modelo = LIBRASClassifier(num_classes=num_classes)
    modelo.load_state_dict(torch.load(caminho_pesos, map_location=device))
    modelo.to(device)
    modelo.eval()
    return modelo
