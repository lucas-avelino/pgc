#!/usr/bin/env bash
# Roda o tools/bench_matrix.py com a rede de teste num namespace de rede proprio.
#
# Por que: com network_mode host, todo o trafego entre os nos passa pela tabela
# de conntrack do host (262144 entradas no beluga). Com ~700+ nos ela enche em
# segundos e o kernel passa a descartar pacotes de pares novos (timeouts de 5 s
# no kademlia). Num namespace criado com --network none, sem regras de iptables,
# o kernel nao rastreia as conexoes (medido: 320 mil pares UDP, contador em 0).
#
# Como:
#   1. constroi a imagem dos nos (mesmo Dockerfile) como $PGC_BENCH_IMAGE;
#   2. sobe um container "dono" do namespace ($PGC_NETNS, --network none);
#   3. roda o bench_matrix.py num container que entra nesse namespace, com o
#      socket e o CLI do Docker do host (o bench continua controlando o compose),
#      --pid host e o cgroup do host (o monitor de CPU/memoria ve os nos como
#      hoje), o repositorio montado no mesmo caminho e o seu usuario (os
#      resultados ficam com o seu dono);
#   4. o bench gera o override com network_mode "container:$PGC_NETNS" para o
#      bootstrap e os servers.
#
# Uso (os argumentos vao direto para o bench_matrix.py):
#   tools/bench_netns.sh --nodes 1023 --id-scheme equal --queries 1000
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT="$(dirname "$REPO")"
NETNS="${PGC_NETNS:-pgc-netns}"
IMAGE="${PGC_BENCH_IMAGE:-pgc-bench}"

DOCKER_BIN="$(command -v docker)"
COMPOSE_PLUGIN=""
for dir in "$HOME/.docker/cli-plugins" /usr/local/lib/docker/cli-plugins /usr/local/libexec/docker/cli-plugins \
           /usr/lib/docker/cli-plugins /usr/libexec/docker/cli-plugins; do
    if [ -x "$dir/docker-compose" ]; then COMPOSE_PLUGIN="$dir/docker-compose"; break; fi
done
if [ -z "$COMPOSE_PLUGIN" ]; then
    echo "nao achei o plugin docker-compose (docker compose v2) do host" >&2
    exit 1
fi
SOCKET=/var/run/docker.sock
SOCKET_GID="$(stat -c %g "$SOCKET")"

echo "[bench_netns] construindo a imagem $IMAGE"
docker build -q -t "$IMAGE" "$REPO" >/dev/null

if docker ps -a --format '{{.Names}}' | grep -qx "$NETNS"; then
    echo "[bench_netns] o container $NETNS ja existe; remova-o (docker rm -f $NETNS) ou escolha outro PGC_NETNS" >&2
    exit 1
fi
echo "[bench_netns] criando o namespace de rede (container $NETNS, --network none)"
docker run -d --name "$NETNS" --network none "$IMAGE" sleep infinity >/dev/null
cleanup() {
    echo "[bench_netns] removendo o container $NETNS"
    docker rm -f "$NETNS" >/dev/null 2>&1 || true
}
trap cleanup EXIT

TTY=()
[ -t 0 ] && [ -t 1 ] && TTY=(-it)

common=(
    --rm
    --network "container:$NETNS"
    --pid host --cgroupns host
    -v /sys/fs/cgroup:/sys/fs/cgroup:ro
    -v "$SOCKET:$SOCKET"
    -v "$DOCKER_BIN:/usr/local/bin/docker:ro"
    -v "$COMPOSE_PLUGIN:/usr/local/lib/docker/cli-plugins/docker-compose:ro"
    -v "$PARENT:$PARENT"
    -w "$REPO"
    --user "$(id -u):$(id -g)" --group-add "$SOCKET_GID"
    -e HOME=/tmp
    -e PYTHONPATH="$REPO/kademlia-lib:$REPO"
    --ulimit nofile=1048576:1048576
)

# confere, antes da bateria, que o CLI do Docker funciona dentro do container
if ! docker run "${common[@]}" "$IMAGE" docker compose version >/dev/null 2>&1; then
    echo "[bench_netns] o 'docker compose' do host nao roda dentro da imagem $IMAGE" >&2
    echo "  (binario do docker dinamico/incompativel?) -- teste: docker run --rm -v $DOCKER_BIN:/usr/local/bin/docker:ro $IMAGE docker version" >&2
    exit 1
fi

echo "[bench_netns] rodando o bench dentro do namespace $NETNS"
docker run "${common[@]}" "${TTY[@]}" "$IMAGE" python3 tools/bench_matrix.py --netns "$NETNS" "$@"
