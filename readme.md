# Estrutura
- core
  - server `onde fica maior parte do código relacionado ao servidor`
  - shared `código compartilhado entre client e nós da rede`
  - client `código do cli`
- kademlia-lib        `clone do repo bmuller/kademlia com as modificaçõees necessarias no projeto`
- embedder            `script que gera o embedding das imagens do projeto`
- persist             `contem pastas com a persistência dos nós, serve para hot start da rede, sem precisar fazer fill da rede`
- Dockerfile          `arquivo de build do container docker`
- docker-compose.yml  `arquivo de configuração do docker compose`


# HOW TOs







get_near_files ../embeddings-1024/000000000009.jpg.bin 64 12

fill ../embeddings-1024/ 1892640

retry_failed fill_failed_requests.json

performance ../embeddings-1024/ 100 2

performance ../embeddings-1024/ 5914 1
000000000009

