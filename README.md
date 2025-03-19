# Video-RAG: Enterprise-Grade Multimodal Retrieval Augmented Generation System

A powerful system for efficient information retrieval from diverse image and video sources using advanced embedding and RAG techniques.

## Features

- Multimodal processing (text and images)
- Efficient vector storage with Milvus
- Advanced embedding generation with Nomic
- Context-aware response generation with Mistral
- Comprehensive logging and error handling
- Batch processing support
- Command-line interface

## Prerequisites

- Python 3.8+
- Docker (for running Milvus)
- CUDA-capable GPU (recommended for better performance)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Video-RAG.git
cd Video-RAG
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start Milvus using Docker:
```bash
docker-compose up -d
```

## Configuration

1. Copy the example configuration:
```bash
cp config/example.yaml config/default.yaml
```

2. Edit `config/default.yaml` with your settings:
```yaml
embedding:
  model: "nomic-embed-text-v1"
  batch_size: 32
  max_length: 512
  image_size: [224, 224]
  output_dir: "data/embeddings"

vector_store:
  host: "localhost"
  port: 19530
  alias: "default"
  dimension: 768

text_collection: "text_embeddings"
image_collection: "image_embeddings"
model: "mistralai/Mistral-7B-Instruct-v0.2"

logging:
  level: "INFO"
  format: "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
  file: "logs/app.log"
```

## Usage

### Adding Content to Knowledge Base

1. Add text content:
```bash
python -m src.rag.run_pipeline add path/to/text/file.txt --modality text
```

2. Add image content:
```bash
python -m src.rag.run_pipeline add path/to/image/directory --modality image
```

3. Add content with metadata:
```bash
python -m src.rag.run_pipeline add path/to/content --metadata path/to/metadata.json
```

### Querying the Knowledge Base

1. Text query:
```bash
python -m src.rag.run_pipeline query "your question here" --modality text
```

2. Image query:
```bash
python -m src.rag.run_pipeline query path/to/query/image.jpg --modality image
```

3. Query with custom parameters:
```bash
python -m src.rag.run_pipeline query "your question" --modality text --top-k 3
```

### Removing Content

```bash
python -m src.rag.run_pipeline remove content_id1 content_id2
```

## Project Structure

```
Video-RAG/
├── config/
│   ├── default.yaml
│   └── example.yaml
├── data/
│   ├── embeddings/
│   └── output/
├── logs/
├── src/
│   ├── data_processing/
│   ├── embedding/
│   ├── rag/
│   ├── utils/
│   └── vector_store/
├── tests/
├── requirements.txt
└── README.md
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
black src/ tests/
isort src/ tests/
```

## Troubleshooting

1. **Milvus Connection Issues**
   - Ensure Milvus is running: `docker ps`
   - Check Milvus logs: `docker logs milvus-standalone`

2. **CUDA/GPU Issues**
   - Verify CUDA installation: `nvidia-smi`
   - Check PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`

3. **Memory Issues**
   - Adjust batch size in config
   - Use smaller models
   - Enable memory-efficient attention

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 