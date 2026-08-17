#!/usr/bin/env bash
# Скачивает тела корпусов из объектного хранилища и сверяет хэши.
#
# Тела корпусов лежат во внутреннем хранилище, совместимом с S3. Его адрес в
# репозитории не записан: он задаётся локальной настройкой вне git. Настройка
# берётся из переменной среды или из файла в
# каталоге ~/.config/reasoning-data, который лежит вне git:
#
#   S3_ENDPOINT      ~/.config/reasoning-data/endpoint
#       имя хранилища так, как его знает клиент mcli, то есть начало пути объекта
#   DATA_RELAY_HOST  ~/.config/reasoning-data/relay
#       узел-посредник, необязательный
#
# Хранилище внутреннее и с рабочей машины напрямую видно не всегда. Скрипт
# сначала пробует локального клиента, а если хранилище не отвечает, идёт через
# узел-посредник: там клиент кладёт архив во временный каталог, оттуда архив
# забирается по scp. Посредник только передаёт файлы и ничего не считает.
#
# Что делает скрипт для каждого манифеста из data/manifests:
#   1. скачивает архив, если его ещё нет в data/archives;
#   2. сверяет sha256 архива с манифестом;
#   3. распаковывает в data/corpora;
#   4. сверяет sha256 каждого файла из манифеста.
#
# Запуск:
#   make fetch                     все корпуса
#   bash data/fetch.sh <корпус>    один корпус по имени файла манифеста без .json
#
# Переменные среды:
#   S3_ENDPOINT      имя хранилища для клиента
#   DATA_RELAY_HOST  узел-посредник
#   RELAY_STAGE      временный каталог на посреднике
#   FORCE=1          перекачать, даже если архив уже лежит локально
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFESTS="$ROOT/data/manifests"
ARCHIVES="$ROOT/data/archives"
CORPORA="$ROOT/data/corpora"
SETTINGS="${REASONING_DATA_SETTINGS:-$HOME/.config/reasoning-data}"

log()  { echo "fetch: $*"; }
die()  { echo "fetch: ошибка: $*" >&2; exit 1; }

# Настройка: сначала переменная среды, потом файл вне git.
setting() {  # setting <имя переменной> <имя файла> [обязательна]
    local from_env="${!1:-}" file="$SETTINGS/$2"
    if [ -n "$from_env" ]; then echo "$from_env"; return; fi
    if [ -f "$file" ]; then head -n1 "$file" | tr -d '[:space:]'; return; fi
    if [ "${3:-}" = "обязательна" ]; then
        die "не задано $1. Записать значение в $file или передать переменной среды $1.
  Адрес внутреннего хранилища в репозитории не хранится."
    fi
}

ENDPOINT="$(setting S3_ENDPOINT endpoint обязательна)"
RELAY_HOST="$(setting DATA_RELAY_HOST relay)"
RELAY_STAGE="${RELAY_STAGE:-.cache/reasoning_clean_fetch}"
FORCE="${FORCE:-0}"

PY="${PY:-python3}"

# Значение поля манифеста. Читаем питоном, чтобы не заводить зависимость от jq.
field() {
    "$PY" - "$1" "$2" <<'PYEOF'
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
node = manifest
for key in sys.argv[2].split("."):
    node = node[key]
print(node)
PYEOF
}

# Как ходить в хранилище: напрямую или через узел-посредник.
choose_transport() {
    if command -v mcli >/dev/null 2>&1 && mcli ls "$ENDPOINT/gpb-reasoning" >/dev/null 2>&1; then
        echo прямой
        return
    fi
    if [ -n "$RELAY_HOST" ] && ssh -o BatchMode=yes -o ConnectTimeout=20 "$RELAY_HOST" \
           "mcli ls '$ENDPOINT/gpb-reasoning' >/dev/null 2>&1"; then
        echo посредник
        return
    fi
    die "хранилище недоступно ни напрямую, ни через посредника.
  Проверить настройку клиента mcli и значения S3_ENDPOINT и DATA_RELAY_HOST в $SETTINGS."
}

download() {  # download <объект> <куда>
    local object="$1" target="$2"
    case "$TRANSPORT" in
        прямой)
            log "качаю напрямую: $object"
            mcli cp "$ENDPOINT/$object" "$target"
            ;;
        посредник)
            local name; name="$(basename "$object")"
            log "качаю через посредника: $object"
            ssh "$RELAY_HOST" "mkdir -p '$RELAY_STAGE' && mcli cp --quiet '$ENDPOINT/$object' '$RELAY_STAGE/$name'" \
                || die "не удалось положить объект во временный каталог на посреднике"
            scp -q "$RELAY_HOST:$RELAY_STAGE/$name" "$target" \
                || die "не удалось забрать архив с посредника"
            ssh "$RELAY_HOST" "rm -f '$RELAY_STAGE/$name'" || true
            ;;
    esac
}

verify_archive() {  # verify_archive <файл> <ожидаемый sha256>
    local file="$1" expected="$2"
    local actual; actual="$(sha256sum "$file" | cut -d' ' -f1)"
    [ "$actual" = "$expected" ] || die "хэш архива не совпал: $file
  ожидалось: $expected
  получено:  $actual"
    log "хэш архива совпал: $(basename "$file")"
}

verify_files() {  # verify_files <манифест> <корень распаковки>
    "$PY" - "$1" "$2" <<'PYEOF'
import hashlib, json, sys
from pathlib import Path

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
root = Path(sys.argv[2])
bad, missing = [], []
for entry in manifest["файлы"]:
    path = root / entry["путь"]
    if not path.is_file():
        missing.append(str(path))
        continue
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != entry["sha256"]:
        bad.append(str(path))
    elif path.stat().st_size != entry["размер"]:
        bad.append(str(path) + " (размер)")

if missing or bad:
    for path in missing:
        print("нет файла: " + path, file=sys.stderr)
    for path in bad:
        print("хэш не совпал: " + path, file=sys.stderr)
    sys.exit(1)
print(f"сверено файлов: {len(manifest['файлы'])}, все хэши совпали")
PYEOF
}

fetch_one() {  # fetch_one <манифест>
    local manifest="$1"
    local stem; stem="$(basename "$manifest" .json)"
    local object; object="$(field "$manifest" 'архив.объект')"
    local name;   name="$(field "$manifest" 'архив.имя')"
    local sha;    sha="$(field "$manifest" 'архив.sha256')"
    local root;   root="$(field "$manifest" 'архив.корень_распаковки')"

    log "корпус $stem"
    mkdir -p "$ARCHIVES" "$CORPORA"
    local archive="$ARCHIVES/$name"

    if [ -f "$archive" ] && [ "$FORCE" != "1" ]; then
        log "архив уже есть локально: $archive"
    else
        download "$object" "$archive"
    fi
    verify_archive "$archive" "$sha"

    rm -rf "${CORPORA:?}/$stem.incoming"
    mkdir -p "$CORPORA/$stem.incoming"
    log "распаковываю"
    tar --use-compress-program=unzstd -xf "$archive" -C "$CORPORA/$stem.incoming"

    local unpacked="$CORPORA/$stem.incoming/$root"
    [ -d "$unpacked" ] || die "в архиве нет ожидаемого корня распаковки: $root"

    log "сверяю хэши файлов"
    verify_files "$manifest" "$unpacked" || die "состав корпуса $stem не совпал с манифестом"

    rm -rf "${CORPORA:?}/$stem"
    mv "$unpacked" "$CORPORA/$stem"
    rm -rf "$CORPORA/$stem.incoming"
    log "готово: $CORPORA/$stem"
}

main() {
    [ -d "$MANIFESTS" ] || die "нет каталога манифестов: $MANIFESTS"
    local list=()
    if [ "$#" -gt 0 ]; then
        for name in "$@"; do
            local path="$MANIFESTS/$name.json"
            [ -f "$path" ] || die "манифеста нет: $path"
            list+=("$path")
        done
    else
        while IFS= read -r path; do list+=("$path"); done \
            < <(find "$MANIFESTS" -maxdepth 1 -name '*.json' | sort)
    fi

    if [ "${#list[@]}" -eq 0 ]; then
        log "манифестов нет, скачивать нечего"
        return 0
    fi

    command -v unzstd >/dev/null 2>&1 || die "нет распаковщика unzstd, поставить пакет zstd"

    TRANSPORT="$(choose_transport)"
    log "способ доступа к хранилищу: $TRANSPORT"

    for manifest in "${list[@]}"; do
        fetch_one "$manifest"
    done
    log "скачано корпусов: ${#list[@]}"
}

main "$@"
