# ortopol-kb — база знаний по работам И. И. Шарапудинова

Инструменты корпуса работ И. И. Шарапудинова (68 источников, ~2500 страниц):
Postgres 17 + pgvector, полнотекстовый поиск с русским стеммингом и семантический
поиск по векторам bge-m3. В git — код; данные приходят артефактом (релиз-ассет).

Рядом с корпусом база держит **внешнюю литературу** — работы чужих авторов,
на которые опирается исследование (`theory/external/`, класс
`external-literature`). Она ищется теми же двумя ключами и **никогда не входит
в публичный артефакт**: чужой копирайт из этой базы не раздаётся.

## Развернуть базу

Что нужно на хосте (ничего сверх этого): Docker с `docker compose` v2; `python3` ≥ 3.10
(только стандартная библиотека — ни одного pip-пакета); клиент `psql` (все скрипты ходят
в базу через него, драйвера нет); `zstd` для распаковки. GPU не обязателен — ollama
считает bge-m3 на CPU (медленнее; включение GPU — закомментированный блок в
`docker-compose.yml`). При первом старте стек тянет образы и модель bge-m3 (~1.2 ГБ) и
собирает образ `kb-pg` (Postgres 17 + pgvector + Apache AGE) из вложенного `pg/Dockerfile`.

Артефакт — в [Releases](https://github.com/sharapudinov/ortopol-kb/releases) этого
репозитория (sha256 — в нотах релиза).

```bash
tar --zstd -xf kb-public-<дата>.tar.zst -C kb-current
cd kb-current
cp .pgenv.example .pgenv        # заполнить PGPASSWORD
docker compose --env-file .pgenv -p kb up -d
python3 profile_checks.py       # состав соответствует манифесту
python3 smoke_test.py           # самопроверка развёртывания
```

Артефакт самодостаточен, этот репозиторий для развёртывания не нужен. Гайд по
работе с развёрнутой базой — `deploy/AGENT_GUIDE.md` внутри артефакта.

## Искать

```bash
set -a; . .pgenv; set +a
python3 pg_search.py "повторные средние Валле-Пуссена" --mode hybrid
```

Режимы: `fulltext` (стемминг: «повторных средних» находит «повторные средние»),
`vector` (по смыслу, без совпадения слов), `hybrid`.

## Раскладка

| | |
| --- | --- |
| `pg_schema.sql` | схема: `corpus.documents` / `corpus.pages` / `corpus.embedding_model` |
| `build_corpus.py`, `pdf_extract.py`, `encoding.py`, `report.py` | извлечение текста из PDF, классификация качества извлечения, отчёт о покрытии |
| `pg_load*.py`, `pg_source_urls.py`, `pg_embed.py` | загрузчики и эмбеддинги |
| `external_registry.py`, `pg_load_external.py`, `external_checks.py` | внешняя литература: реестр `theory/external/EXTERNAL_INDEX.md`, её загрузчик и её проверки |
| `pg_search.py` | поиск |
| `pg_schema_citation*.sql`, `citation_vocab.py`, `pg_graph.py`, `pg_graph_candidates.py`, `pg_graph_cocitation.py`, `pg_graph_cypher.py` | граф цитирований: схема и её словари, AGE-проекция (`citation_graph`), потребители (`citers`/`candidates`/`cocitation`/`hybrid`) |
| `corpus_completeness.py` | предикат полноты корпуса |
| `deploy/` | сборка артефакта, docker-compose стек, смок-тест, проверки состава |
| `EXTENDING.md` | процедуры пополнения базы |
| `deploy/AGENT_GUIDE.md` | гайд получателя артефакта |

## Собрать артефакт

```bash
set -a; . ../corpus/.pgenv; set +a
python3 deploy/build_package.py --profile public
```

Результат — `../corpus/deploy/kb-public-<дата>.tar.zst`. Состав каждого документа
определяет колонка `public_distribution` в `corpus.documents`:

| значение | что в артефакте |
| --- | --- |
| `full-text` | текст, векторы, исходный PDF |
| `internal` | то же (служебные документы INDEX/THEMES) |
| `metadata-only` | библиография, правовые колонки, `source_sha256`, ссылка на оригинал; у страниц — векторы |
| `excluded` | ничего: ни строки документа, ни страниц, ни векторов |

`excluded` — для документов, чей правовой режим владелец не установил (выложить даже
библиографию значило бы решить открытый вопрос по умолчанию), и для всей внешней
литературы: работа чужого автора наружу не идёт ни в каком виде. Манифест при этом
перечисляет весь корпус (`legal.documents_by_distribution`) и называет вошедшие в
пакет классы (`legal.shipped_distributions`), так что исключённое видно поимённо.

Режим схемы `citation` в публичном пакете — строка `citation.public_policy`, и
манифест говорит не только КАКОЙ он, но и ЧЕЙ: `citation.policy_source` = `owner`
(прочитано из базы) либо `override` (задано флагом `--policy-override`, только для
прогона конвейера). На `override` `profile_checks.py` отказывает: такой артефакт
не публикуется, как бы он ни назывался. Имя у него тоже своё —
`kb-override-<профиль>-<дата>.tar.zst`, вне пространства `kb-public-*`, так что
отбор по имени профиля до него не дотягивается.

Проверка состава по байтам артефакта:
`python3 deploy/profile_checks.py --artifact-dir <распакованный>` → exit 0.

## Тесты

```bash
python3 -m unittest discover -s tests -t tests
```

1405 тестов, без Docker и без сети; тестам, которым нужна живая база, при её
отсутствии положено пропускаться с внятным сообщением.
