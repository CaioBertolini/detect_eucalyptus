# Detecção de Eucaliptos

Este repositório contém um script em Python para detectar brotos de eucalipto em imagens usando um modelo YOLO treinado. O script processa imagens TIFF, realiza detecções, calcula métricas de distribuição das plantas e salva os resultados em formato GeoJSON e JSON.

## Pré-requisitos

- Python 3.8 ou superior
- Git (para clonar o repositório)

## Instalação

### 1. Clonagem do Repositório

Clone este repositório para sua máquina local:

```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
```

### 2. Criação de Ambiente Virtual

É recomendado criar um ambiente virtual para isolar as dependências do projeto:

```bash
python -m venv env
source env/bin/activate  # No Windows: env\Scripts\activate
```

### 3. Instalação das Dependências

Instale as bibliotecas necessárias usando o arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

As dependências incluem:
- `ultralytics`: Para usar o modelo YOLO
- `opencv-python`: Para processamento de imagens
- `rasterio`: Para leitura de arquivos TIFF georreferenciados
- `numpy`: Para operações numéricas
- `geopandas`: Para manipulação de dados geoespaciais
- `shapely`: Para geometrias espaciais

## Como Executar o Script

O script principal é `detect_eucalyptus.py`. Ele aceita os seguintes argumentos:

- `--source`: Caminho para o arquivo de imagem TIFF (obrigatório)
- `--output`: Diretório para salvar os resultados (opcional - padrão: 'results')
- `--model`: Caminho para o arquivo do modelo YOLO (.pt) (opcional - padrão: './runs/detect/eucalipto/v5/weights/best.pt')
- `--conf`: Limite de confiança para detecção (opcional - padrão: 0.5)
- `--gsd`: Distância de amostragem do solo (GSD) em metros por pixel (opcional - padrão: 0.05)

### Exemplo de Execução

```bash
python detect_eucalyptus.py --source caminho/para/imagem.tif
```

### Saídas

O script gera dois arquivos no diretório de saída:
- `detections.geojson`: Vetores geográficos das detecções em formato GeoJSON
- `metrics.json`: Métricas calculadas, incluindo número total de detecções, área total, detecções por hectare, desvio padrão, índice de Gini e PV50

## Estrutura do Projeto

```
.
├── detect_eucalyptus.py    # Script principal
├── requirements.txt        # Dependências do Python
├── runs/
│   └── detect/
│       └── eucalipto/
│           └── v6/
│               ├── args.yaml
│               ├── results.csv
│               └── weights/
│                   ├── best.pt    # Modelo YOLO treinado
│                   └── last.pt
└── README.md               # Este arquivo
```

## Notas

- Certifique-se de que a imagem de entrada seja um arquivo TIFF válido com coordenadas geográficas.
- O modelo YOLO deve estar no formato .pt e compatível com a versão do Ultralytics usada.
- As métricas de Gini e PV50 são calculadas com base nas áreas das detecções para avaliar a homogeneização do plantio.

Para mais informações ou suporte, entre em contato com o mantenedor do repositório.