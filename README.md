# ProgettoAI 81940

Un agente di Reinforcement Learning che impara a guidare un kart su circuiti di gara usando Deep Q-Network (DQN).

## Obiettivo

Addestrare un agente RL a:
- Navigare autonomamente su un circuito
- Attraversare tutti i checkpoint in ordine
- Evitare di uscire fuori pista
- Completare il giro nel minor tempo possibile

## Architettura

```
src/
├── main.py              # Entry point (train/test)
├── agents/
│   ├── agent.py         # DQN Agent + ReplayBuffer
│   └── network.py       # Neural Network (MLP)
├── env/
│   └── track_env.py     # Ambiente Gymnasium
└── utils/
    └── visual.py        # Grafici e heatmap
```

## Quick Start

### Installazione

```bash
# Clona il repository
git clone https://github.com/EChiarini/ProgettoAI.git
cd ProgettoAI

# Installa le dipendenze con uv
uv sync
```

### Training

```bash
# Training con 1000 episodi (default)
python src/main.py train

# Training con numero episodi personalizzato
python src/main.py train --ep 5000
```

### Testing

```bash
# Test con il miglior modello
python src/main.py test

# Test con checkpoint specifico
python src/main.py test --mod cp_5000.pth
```

## Output

Dopo il training vengono generati in `results/`:
- `grafico_finale.png` - Andamento reward
- `heatmap_finale.png` - Traiettorie percorse
- `reports/report_N.pdf` - Report automatico

## Documentazione

- Relazione completa: `docs/Relazione.ipynb`
