# Pesquisa Semântica de Imagens em Tabelas de Hash Distribuídas

> Antes de clonar o repositório, é importante mecionar que a pasta top500 pesa 1gb em disco.

## Estrutura
- core
  - server `onde fica maior parte do código relacionado ao servidor`
  - shared `código compartilhado entre client e nós da rede`
  - client `código do cli`
- kademlia-lib        `clone do repo bmuller/kademlia com as modificaçõees necessarias no projeto`
- embedder            `script que gera o embedding das imagens do projeto`
- tools               `scripts auxiliares de analise, fora do caminho da rede`
- data
  - top500            `Os 500 vizinhos mais proximos de cada embedding, um json por imagem, Essa pasta contem`
- persist             `contém pastas com a persistência dos nós, serve para hot start da rede, sem precisar fazer fill da rede`
- Dockerfile          `arquivo de build do container docker`
- docker-compose.yml  `arquivo de configuração do docker compose`

## Setup CLI
Certifique-se que está em um venv valido.
- `pip install -r requirements.txt`
- `pip install -r server-requirements.txt`

## Como subir a rede
Para rodar a rede das duas formas possiveis é necessario docker e docker compose instalado na maquina.
Para rodar o cli é necessario python instalado. 

### Rodando a rede com imagens já na memoria
O repositório ja possui a memoria de containers com as imagens do dataset COCO, para rodar buscas dentro desse dataset é necessario apenas, rodar o seguinte commando:
- `docker compose up -d --build`
Após a rede subir o cli pode ser aberto com o seguinte comando:
windows:
- `$env:SERVER_COUNT = "128"; python -m core.cli`
linux:
- `SERVER_COUNT="128" && python -m core.cli`

> 128 aqui significa a quantidade de nós que o CLI vai conhecer da rede

### Rede sem persistencia
Só é necessario um passo antes dos demais, que seria apagar a pasta persist no repositório, os nós ja identificam que não tem slot de memoria pra eles e cria um do zero, escolhendo um node ID para eles aleatóriamente

## Comandos CLI
- `get_near_files ./data/embeddings-1024/000000000009.jpg.bin 64 12`
  - pesquisa por todas as chaves do arquivo `./data/embeddings-1024/000000000009.jpg.bin` retornando a lista de arquivos que possuem chaves com um raio de no minimo 12 bits, o parametro 64 é o tamanho da chave
  
- `performance ./data/embeddings-1024/ 100`
  - executa 100 requests aleatórias com embeddings na pasta `./data/embeddings-1024/` e mede performance de tempo das requests

- `recall_rand ./data/embeddings-1024 100 12`
  - calcula o recall para 100 imagens com raio 12

- `recall ./data/embeddings-1024/000000399664.jpg.bin 12`
  - calcula o recall para o embedding da imagem `./data/embeddings-1024/000000399664.jpg.bin` com raio 12

- `fill ./data/embeddings-1024/ 1892640` 
  - esse comando serve para popular a rede com os emveddings da pasta, 1892640 é a quantidade de embeddings que ele vai cadastrar até parar, no caso, representa o numero total, esse comando demora por volta de 1h50 para os embeddings padrões do projeto
