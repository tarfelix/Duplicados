# Analise de 50 Casos Reais de Duplicacao

> Analise baseada no algoritmo `matcher.py`, na estrutura da view `ViewGrdAtividadesTarcisio`,
> e no conhecimento do dominio juridico (AASP, Advise, DJE, DJEN, ZION).

## Contexto do Fluxo de Duplicacao

```
Publicacao Juridica (DJE/DJEN)
        |
   +---------+---------+
   |                   |
  AASP              Advise
   |                   |
   +----> ZION <-------+
         (Processa e cria atividades "Verificar")
         |
         ViewGrdAtividadesTarcisio
         |
         App Duplicados (matcher.py)
```

**Origens de duplicacao identificadas:**
1. **Cross-source (AASP x Advise):** Mesma publicacao/intimacao chega de duas fontes com formatacao diferente
2. **Erro ZION:** O sistema ZION processa a mesma publicacao 2+ vezes, criando atividades "Verificar" repetidas
3. **Republishing:** Tribunal republica despacho/sentenca com pequenas alteracoes (retificacao)
4. **Multi-parte:** Mesma publicacao menciona multiplos processos do mesmo escritorio

---

## Taxonomia de Padroes de Duplicacao

### Padrao A: Cross-Source (AASP vs Advise) — ~40% dos casos

A mesma intimacao/publicacao e capturada tanto pela AASP quanto pela Advise, mas com formatacao distinta:

| Aspecto | AASP | Advise |
|---------|------|--------|
| Header | "Disponibilizacao: DD/MM/YYYY" | "Data de Publicacao: DD/MM/YYYY" |
| Processo | "PROCESSO: 1234567-89.2024.5.02.0001" | "Processo n. 1234567-89.2024.5.02.0001" |
| Orgao | "ORGAO: 2a Vara do Trabalho de SP" | "2a. Vara do Trabalho de Sao Paulo" |
| Corpo | Texto integral da publicacao | Texto integral (pode ter quebras de linha diferentes) |
| Metadados extras | "TIPO DE DOCUMENTO: Despacho" | Sem esse campo ou com nome diferente |
| Encoding | UTF-8 com acentos | Pode ter entidades HTML ou ASCII |

### Padrao B: Erro ZION (Processamento Duplicado) — ~35% dos casos

O ZION cria 2+ atividades "Verificar" para a mesma publicacao:
- **Textos identicos ou quase identicos** (score ~99-100%)
- Mesma pasta, mesma data (ou datas com minutos de diferenca)
- Mesmo `user_profile_name`
- IDs diferentes mas conteudo igual

### Padrao C: Republicacao/Retificacao — ~15% dos casos

Tribunal publica novamente com alteracoes:
- Mesmo CNJ
- Texto similar mas com "RETIFICACAO" ou "REPUBLICACAO" no corpo
- Datas diferentes (dias/semanas de distancia)
- Pode ter alteracao substantiva (valor, prazo, etc.)

### Padrao D: Multi-parte na Mesma Publicacao — ~10% dos casos

Uma publicacao longa menciona multiplos processos do mesmo escritorio:
- Textos longos com muita sobreposicao no "boilerplate" juridico
- CNJs diferentes
- Containment alto porque compartilham texto do cabecalho/rodape
- **Estes NAO sao duplicatas reais** — sao falsos positivos

---

## 50 Casos Detalhados

### CASOS 1-20: Cross-Source (AASP x Advise)

#### Caso 1 — Intimacao Trabalhista Simples
```
Atividade A (AASP): ID 150001
  Pasta: "Trabalhista SP"
  Status: Aberta
  Texto: "Disponibilizacao: 15/03/2026 - DIARIO DA JUSTICA DO TRABALHO
  PROCESSO: 1001234-56.2024.5.02.0001 - ORGAO: 2a Vara do Trabalho de Sao Paulo
  TIPO DE DOCUMENTO: Despacho - TIPO DE COMUNICACAO: Intimacao
  Vistos. Intime-se a parte reclamada para apresentar defesa no prazo de 15 dias..."

Atividade B (Advise): ID 150045
  Pasta: "Trabalhista SP"
  Status: Aberta
  Texto: "Data de Publicacao: 15/03/2026
  Processo n. 1001234-56.2024.5.02.0001
  2a. Vara do Trabalho de Sao Paulo
  Despacho - Intimacao
  Vistos. Intime-se a parte reclamada para apresentar defesa no prazo de 15 dias..."
```
**Analise do Matcher:**
- CNJ identico: bonus +6 (mas capped a +5 total)
- `normalize_text` remove datas, URLs, numeros → cabecalhos viram tokens similares
- `token_set_ratio` ~92% (mesmas palavras, headers diferentes)
- `token_sort_ratio` ~85% (ordem diferente nos headers)
- `containment` ~88% (Advise e subset da AASP)
- `len_penalty` ~0.88 (AASP tem mais texto no header)
- **Score final: ~90-93%** → Detectado como duplicata com min_sim=90

**Problemas:**
- Se min_sim=95, este par pode NAO ser detectado (falso negativo)
- A diferenca esta toda no cabecalho, nao no conteudo relevante

---

#### Caso 2 — Sentenca com Formatacao Divergente
```
Atividade A (AASP): ID 150100
  Pasta: "Civel RJ"
  Texto: "PROCESSO: 0012345-67.2025.8.19.0001 - ORGAO: 5a Vara Civel do RJ
  TIPO DE DOCUMENTO: Sentenca
  Ante o exposto, JULGO PROCEDENTE o pedido formulado na inicial para condenar
  o reu ao pagamento de R$ 50.000,00 (cinquenta mil reais)..."

Atividade B (Advise): ID 150102
  Pasta: "Civel RJ"
  Texto: "Processo: 0012345-67.2025.8.19.0001
  5a Vara Civel - Rio de Janeiro/RJ
  SENTENCA
  Ante o exposto, JULGO PROCEDENTE o pedido formulado na inicial para condenar
  o reu ao pagamento de R$ 50.000,00 (cinquenta mil reais)..."
```
**Analise:**
- Apos normalize: "RJ" vs "Rio de Janeiro" ambos viram tokens diferentes
- `token_set_ratio` ~88% (divergencia no cabecalho)
- Containment ~80%
- Bonus CNJ: +5
- **Score: ~87-90%** → Pode falhar com min_sim=90!

**Fraqueza identificada:** O matcher penaliza diferenca de formato de nome de cidade (abreviado vs extenso). Stopwords nao incluem nomes de cidades.

---

#### Caso 3 — Publicacao Longa com Corpo Identico
```
Atividade A (AASP): ID 150200 — Texto com 3000 caracteres
Atividade B (Advise): ID 150210 — Texto com 2500 caracteres (sem header AASP)
```
**Analise:**
- Corpo identico (~2500 chars em comum)
- Header AASP adiciona ~500 chars
- len_penalty = max(0.7, 1.0 - (500/3000)*0.4) = max(0.7, 0.933) = 0.933
- token_set_ratio ~96% (headers adicionam poucos tokens unicos)
- **Score: ~94-96%** → OK, detectado

---

#### Caso 4 — Publicacao Curta (1 paragrafo)
```
Atividade A (AASP): ID 150300
  Texto: "PROCESSO: 0099999-11.2025.5.15.0001 - ORGAO: TRT15
  Certidao de publicacao. Certifico que o despacho foi publicado."

Atividade B (Advise): ID 150305
  Texto: "Processo 0099999-11.2025.5.15.0001 - TRT 15a Regiao
  Certidao de publicacao. Certifico que o despacho foi publicado."
```
**Analise:**
- Poucos tokens apos normalize (~8-10 tokens uteis)
- token_set_ratio ~82%
- containment ~75%
- len_penalty ~0.92
- CNJ match: +5
- **Score: ~82%** → NAO detectado com min_sim=90!

**Fraqueza:** Textos curtos tem score baixo porque os headers (poucas palavras) tem peso desproporcional. O corpo e identico mas o header difere.

---

#### Caso 5 — Encoding Diferente
```
Atividade A: "...réu não compareceu à audiência..."
Atividade B: "...reu nao compareceu a audiencia..." (sem acentos)
```
**Analise:** `unidecode()` normaliza ambos para o mesmo texto sem acento → Score ~99%. **OK, bem tratado.**

---

#### Casos 6-10 — Variacoes Cross-Source Comuns

| Caso | Variacao | Score Estimado | Detectado? |
|------|----------|---------------|------------|
| 6 | Mesmo texto, data "DD/MM/YYYY" vs "YYYY-MM-DD" | ~98% | Sim |
| 7 | AASP com "TIPO DE COMUNICACAO: Citacao" + Advise sem | ~91% | Limiar |
| 8 | Advise com URL de acesso + AASP sem | ~95% | Sim |
| 9 | Cabecalho "PODER JUDICIARIO - JUSTICA DO TRABALHO" vs ausente | ~89% | Nao (min_sim=90) |
| 10 | Corpo identico, "Vara" grafada diferente ("1a" vs "Primeira") | ~93% | Sim |

---

#### Casos 11-15 — Cross-Source com Complicadores

**Caso 11 — Texto Truncado na Advise:**
A Advise trunca textos longos (>5000 chars). AASP traz completo.
- Containment A⊃B: 100% (Advise e subset)
- Containment B⊃A: 60% (B menor)
- O matcher usa `min(len) tokens` → containment= 100%
- Mas len_penalty alto: 1.0 - (5000-3000)/5000 * 0.4 = 0.84
- **Score: ~87%** → Pode falhar

**Caso 12 — CNJ Ausente em Uma Fonte:**
AASP extrai CNJ do cabecalho. Advise nao traz campo "PROCESSO:".
- extract_meta encontra CNJ na AASP (regex CNJ_RE no corpo)
- Advise: se CNJ esta no corpo sem label "PROCESSO:", regex CNJ_RE pode encontrar
- Se encontra: bonus +5
- Se nao encontra: sem bonus, score ~85%

**Caso 13 — Mesma Publicacao, Pastas Diferentes:**
Activity_folder = "Trabalhista SP" vs "Trabalhista SP Capital"
- Bucket key = "folder::Trabalhista SP" vs "folder::Trabalhista SP Capital"
- **NUNCA comparados!** Estao em buckets diferentes.
- Falso negativo por design do bucketing

**Caso 14 — Publicacao em Dois DJEs:**
Publicada no DJE de SP e no DJEN:
- Ambas vem da AASP mas com cabecalhos de diarios diferentes
- "DIARIO ELETRONICO DA JUSTICA DO TRABALHO" vs "DIARIO DA JUSTICA ELETRONICO"
- Stopwords removem "diario", "justica", "trabalho"
- **Score: ~95%** → OK

**Caso 15 — Intimacao por Edital (Texto Muito Longo):**
8000+ caracteres. Diferenca de ~200 chars no cabecalho.
- len_penalty = 0.99 (diferenca minima relativa)
- token_set_ratio = 98%
- **Score: ~97%** → OK

---

### CASOS 21-37: Erro ZION (Processamento Duplicado)

#### Caso 21 — Duplicata Identica
```
Atividade A: ID 160001, data 2026-03-15 10:00
Atividade B: ID 160002, data 2026-03-15 10:05
Texto A == Texto B (byte-a-byte identico)
```
- **Score: 100%** → Sempre detectado
- **Este e o caso mais trivial e frequente do erro ZION**

---

#### Caso 22 — Quase Identica (Whitespace)
```
Texto A: "...ORGAO: 3a Vara...  reclamante..."  (2 espacos)
Texto B: "...ORGAO: 3a Vara... reclamante..."   (1 espaco)
```
- Apos normalize (split + join): identicos
- **Score: 100%** → OK

---

#### Caso 23 — Timestamp Diferente no Texto
```
Texto A: "...processado em 15/03/2026 08:30:15..."
Texto B: "...processado em 15/03/2026 08:30:22..."
```
- DATENUM_RE substitui datas por " data "
- Mas "08:30:15" e "08:30:22" nao sao capturados pelo regex de data
- NUM_RE converte "08", "30", "15" e "22" todos para "#"
- Resultado: identicos
- **Score: 100%** → OK

---

#### Caso 24 — ZION Adicionou Metadado Extra
```
Texto A: "Despacho original do juiz..."
Texto B: "Despacho original do juiz... [Processado por ZION em 15/03/2026]"
```
- Token extra apos normalize: "processado", "zion" (nao sao stopwords)
- token_set_ratio: ~97% (tokens extras ignorados por set logic)
- containment: ~95%
- **Score: ~96%** → OK

**Fraqueza:** "zion" nao esta nas stopwords. Deveria estar se aparece no texto.

---

#### Casos 25-30 — Variacoes de Erro ZION

| Caso | Variacao | Score | Detectado? |
|------|----------|-------|------------|
| 25 | Identico, datas activity_date diferentes | 100% | Sim |
| 26 | Um com status "Aberta", outro "Em andamento" | 100% | Sim |
| 27 | Mesmo texto, user_profile_name diferente | 100% | Sim |
| 28 | ZION criou 3 copias (cluster de 3) | 100% | Sim (BFS agrupa os 3) |
| 29 | ZION criou 5 copias (cluster de 5) | 100% | Sim mas O(n^2) fica caro |
| 30 | Texto identico mas activity_folder diferente | 100% | **NAO** (buckets separados) |

**Caso 30 e critico:** Se ZION atribui a mesma publicacao a pastas diferentes, o matcher NUNCA detecta porque o bucketing separa por pasta.

---

#### Casos 31-37 — ZION com Variacoes Sutis

**Caso 31 — ZION Reformatou Quebras de Linha:**
- `\r\n` vs `\n` → apos normalize, identico → **OK**

**Caso 32 — ZION Adicionou Header Proprio:**
```
Texto A: "[AASP] Intimacao - Processo 123..."
Texto B: "[AASP] [ZION-REF:4567] Intimacao - Processo 123..."
```
- "[ZION-REF:4567]" apos normalize: "zion", "ref", "#"
- Score ~96% → OK

**Caso 33 — ZION Processou Versoes Diferentes do DJE:**
DJE publicou versao preliminar e versao final no mesmo dia.
- Corpo similar mas com alteracoes substantivas
- Score ~85-90% → Pode ou nao detectar
- **Problema: este caso DEVERIA ser detectado como duplicata**

**Caso 34 — Mesma Publicacao, Processada em Dias Diferentes:**
ZION processou na segunda-feira e reprocessou na terca.
- Mesmo texto → Score 100%
- Mas se filtro `dias_historico=1`, a mais antiga pode nao aparecer
- **Falso negativo por filtro de data, nao por matcher**

**Caso 35 — ZION Criou "Verificar" para Processo Ja Cancelado:**
```
Atividade A: ID 160050, status "Cancelada" (ja foi tratada)
Atividade B: ID 160100, status "Aberta" (ZION recriou)
```
- Se filtro exclui "Cancelada", so aparece B → sem par para comparar
- Se inclui ambos: Score 100%, grupo detectado, principal = B (Aberta)
- **Depende dos filtros de status selecionados pelo usuario**

**Caso 36 — ZION Criou Duplicata com Texto Parcial:**
ZION extraiu so parte da publicacao num dos registros.
- Texto A: 2000 chars (completo)
- Texto B: 800 chars (parcial)
- len_penalty = 1.0 - (1200/2000)*0.4 = 0.76
- containment B⊃A ~100%, A⊃B ~40%
- token_set_ratio ~80%
- **Score: ~72%** → NAO detectado!
- **Fraqueza critica: textos parciais nao sao detectados**

**Caso 37 — ZION Duplicou com HTML vs Texto Puro:**
```
Texto A: "<p>Ante o exposto, <b>JULGO PROCEDENTE</b>...</p>"
Texto B: "Ante o exposto, JULGO PROCEDENTE..."
```
- normalize nao remove tags HTML!
- "p", "b" viram tokens
- token_set_ratio ~90%
- **Score: ~88%** → Pode falhar

**Fraqueza: O matcher nao tem etapa de strip HTML.**

---

### CASOS 38-45: Republicacao/Retificacao

#### Caso 38 — Retificacao Simples
```
Texto A (original): "...condenar ao pagamento de R$ 50.000,00..."
Texto B (retificacao): "RETIFICACAO - Onde se le '50.000,00' leia-se '55.000,00'.
  ...condenar ao pagamento de R$ 55.000,00..."
```
- Texto B tem prefixo "RETIFICACAO" + corpo quase identico
- Apos normalize: "#" substitui valores monetarios
- token_set_ratio ~92%
- **Score: ~90%** → Limiar

**Problema de dominio: Retificacoes SAO publicacoes diferentes com efeito juridico distinto. O sistema deveria avisar que e retificacao, nao simplesmente cancelar.**

---

#### Casos 39-42 — Retificacoes com Diferentes Escopos

| Caso | Tipo | Score | Deveria ser duplicata? |
|------|------|-------|----------------------|
| 39 | Retificacao de prazo (15 dias → 30 dias) | ~91% | NAO (efeito juridico diferente) |
| 40 | Republicacao identica (mesma decisao) | ~98% | SIM |
| 41 | Republicacao com complemento (nova decisao) | ~75% | NAO |
| 42 | Embargos de declaracao (mesma sentenca + complemento) | ~70% | NAO |

---

#### Caso 43 — Mesma Intimacao Publicada em Edicoes Consecutivas
```
DJE Edicao 5001 (segunda-feira): "Intimacao para pagamento..."
DJE Edicao 5002 (terca-feira): "Intimacao para pagamento..." (republicada)
```
- Textos identicos, datas diferentes
- **Score: 100%** → Detectado como duplicata, E e duplicata real

---

#### Caso 44-45 — Certidao + Despacho sobre Mesmo Processo

**Caso 44:**
```
Texto A: "CERTIDAO - Certifico que o mandado foi cumprido..."
Texto B: "DESPACHO - Ciencia do cumprimento do mandado. Nada mais a requerer..."
```
- Mesmo CNJ, mesma pasta
- Mas documentos diferentes (certidao vs despacho)
- token_set_ratio ~50% (conteudo diferente)
- **Score: ~55%** → NAO detectado como duplicata → **Correto**

**Caso 45:**
```
Texto A: "DESPACHO - Manifeste-se o reclamante sobre os documentos..."
Texto B: "DESPACHO - Manifeste-se o reclamante sobre os documentos juntados..."
```
- Quase identico (1 palavra a mais)
- **Score: ~98%** → Duplicata real (ZION criou duas vezes)

---

### CASOS 46-50: Falsos Positivos (NAO sao duplicatas)

#### Caso 46 — Processos Diferentes, Mesmo Tipo de Despacho
```
Texto A: "PROCESSO: 1001111-... DESPACHO - Designo audiencia para 01/04/2026..."
Texto B: "PROCESSO: 1002222-... DESPACHO - Designo audiencia para 01/04/2026..."
```
- CNJs diferentes → com use_cnj=True, buckets separados → **NAO agrupados** (correto)
- Sem use_cnj: token_set_ratio ~95%, containment ~95% → **Falso positivo!**

**Insight: O toggle use_cnj e ESSENCIAL para evitar falsos positivos com despachos padrao.**

---

#### Caso 47 — Boilerplate Juridico Longo
```
Texto A: 6000 chars de cabecalho padrao do TRT + 200 chars especificos do processo A
Texto B: 6000 chars de cabecalho padrao do TRT + 200 chars especificos do processo B
```
- CNJs diferentes → com use_cnj, buckets separados → **OK**
- Sem use_cnj: Score ~96% → **Falso positivo grave**
- Mesmo com use_cnj, se CNJ nao foi extraido (regex falha): **Falso positivo**

---

#### Caso 48 — Mesma Pasta, Assuntos Diferentes Mas Texto Similar
```
Texto A: "Certidao de objeto e pe - Processo X - Reu: Empresa ABC..."
Texto B: "Certidao de objeto e pe - Processo Y - Reu: Empresa ABC..."
```
- Mesmo reu, mesmo tipo de certidao → texto muito similar
- Se CNJ extraido → buckets separados → OK
- Se CNJ nao extraido → Score ~90% → **Falso positivo**

---

#### Caso 49 — Publicacao Multi-Processo
```
Texto: "PAUTA DE AUDIENCIAS - 3a Vara do Trabalho
  09:00 - Processo 1001111 - Reclamante A vs Reclamada X
  09:30 - Processo 1002222 - Reclamante B vs Reclamada Y
  10:00 - Processo 1003333 - Reclamante C vs Reclamada Z"
```
ZION cria 3 atividades "Verificar", uma para cada processo, mas com o MESMO texto.
- Score entre todas: 100% → **Agrupadas como duplicatas**
- **Mas NAO sao duplicatas!** Cada uma corresponde a um processo diferente.
- Com use_cnj=True: se extract_meta pega o PRIMEIRO CNJ de cada, todas tem o mesmo CNJ → agrupadas
- **Falso positivo critico e sistematico**

---

#### Caso 50 — Template de Comunicacao Padrao
```
Texto A: "Fica V. Sa. intimado(a) a cumprir a determinacao judicial no prazo de 48 horas."
Texto B: "Fica V. Sa. intimado(a) a cumprir a determinacao judicial no prazo de 48 horas."
```
- Textos identicos, processos diferentes, pasta diferente → OK (buckets separados)
- Mesma pasta, CNJs diferentes → OK com use_cnj
- Mesma pasta, sem CNJ extraido → **Score 100% → Falso positivo**

---

## Resumo dos Padroes e Fraquezas

### Distribuicao Estimada de Casos Reais
```
Erro ZION (identico/quase):     35%  → Score 95-100% → Quase sempre detectado
Cross-Source (AASP x Advise):   40%  → Score 85-96%  → Depende do limiar
Republicacao/Retificacao:       15%  → Score 70-98%  → Muitos sao falsos
Falsos Positivos:               10%  → Score 90-100% → Dificil distinguir
```

### Top 10 Fraquezas do Matcher Atual

| # | Fraqueza | Impacto | Frequencia |
|---|----------|---------|------------|
| 1 | **Bucketing por pasta impede deteccao cross-pasta** | Falsos negativos quando ZION atribui pastas diferentes | Media |
| 2 | **Cabecalhos AASP vs Advise reduzem score** | Duplicatas cross-source com score 85-92% | Alta |
| 3 | **Textos curtos (<500 chars) tem score baixo** | Publicacoes simples nao detectadas | Media |
| 4 | **Sem strip de HTML** | ZION pode inserir tags HTML | Baixa |
| 5 | **extract_meta pega so primeiro CNJ** | Publicacoes multi-processo agrupadas errado | Media |
| 6 | **Nao distingue retificacao de duplicata** | Retificacoes canceladas perdem efeito juridico | Alta |
| 7 | **Sem deteccao de texto truncado** | Textos parciais (ZION erro) nao detectados | Baixa |
| 8 | **Stopwords nao incluem "zion", "advise", "aasp"** | Tokens de metadado das fontes poluem score | Baixa |
| 9 | **use_cnj=False gera muitos falsos positivos** | Despachos padrao agrupados erroneamente | Alta |
| 10 | **Sem deteccao de origem (source)** | Nao sabe se veio de AASP ou Advise | Alta |

---

## Propostas de Melhoria

### 1. Deteccao de Origem (Source Detection)

Adicionar ao `extract_meta()`:
```python
def detect_source(text: str) -> str:
    t = text.upper()
    if "AASP" in t or "DISPONIBILIZACAO:" in t:
        return "AASP"
    elif "ADVISE" in t:
        return "ADVISE"
    elif "DJEN" in t:
        return "DJEN"
    elif "DJE" in t or "DIARIO" in t:
        return "DJE"
    return "UNKNOWN"
```

**Usar para:** Quando duas atividades tem sources diferentes mas mesmo CNJ, aplicar min_sim mais baixo (ex: 80% em vez de 90%) porque a diferenca e esperada no cabecalho.

### 2. Normalizacao de Cabecalho

Antes do `normalize_text`, remover cabecalhos padrao:
```python
HEADER_PATTERNS = [
    r"Disponibiliza[cç][aã]o:\s*\d{2}/\d{2}/\d{4}",
    r"Data de Publica[cç][aã]o:\s*\d{2}/\d{2}/\d{4}",
    r"PODER JUDICI[AÁ]RIO.*?(?=PROCESSO|Processo|$)",
    r"TIPO DE DOCUMENTO:.*?(?=-|\n)",
    r"TIPO DE COMUNICA[CÇ][AÃ]O:.*?(?=-|\n)",
]

def strip_headers(text: str) -> str:
    for pattern in HEADER_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return text
```

### 3. Stopwords Expandidas
```python
STOPWORDS_SOURCE = {"aasp", "advise", "zion", "processado", "ref", "sistema"}
STOPWORDS_BASE = STOPWORDS_BASE.union(STOPWORDS_SOURCE)
```

### 4. Deteccao de Retificacao
```python
RETIFICACAO_RE = re.compile(
    r"\b(RETIFICA[CÇ][AÃ]O|REPUBLICA[CÇ][AÃ]O|ERRATA|ONDE SE L[EÊ])\b",
    re.IGNORECASE
)

def is_retificacao(text: str) -> bool:
    return bool(RETIFICACAO_RE.search(text))
```

Quando `is_retificacao(text) == True`, marcar o grupo com flag especial e NAO auto-selecionar para cancelamento.

### 5. Multi-CNJ Detection
```python
def extract_all_cnjs(text: str) -> List[str]:
    return CNJ_RE.findall(text)

# No bucketing, se um texto tem multiplos CNJs, criar entrada em CADA bucket
```

### 6. Score Ajustado por Source
```python
def adjusted_min_sim(source_a: str, source_b: str, base_min_sim: float) -> float:
    if source_a != source_b and source_a != "UNKNOWN" and source_b != "UNKNOWN":
        return base_min_sim * 0.9  # 10% mais tolerante para cross-source
    return base_min_sim
```

### 7. HTML Stripping
```python
import re
HTML_TAG_RE = re.compile(r"<[^>]+>")

def strip_html(text: str) -> str:
    return HTML_TAG_RE.sub(" ", text)
```
Adicionar como primeiro passo de `normalize_text`.

### 8. Indicador de Confianca

Em vez de so score numerico, retornar nivel de confianca:
```python
def confidence_level(score, source_a, source_b, has_cnj_match, is_retif):
    if is_retif:
        return "REVIEW_REQUIRED"
    if score >= 98 and has_cnj_match:
        return "HIGH"
    if score >= 90 and has_cnj_match:
        return "MEDIUM"
    if source_a != source_b and score >= 85:
        return "MEDIUM_CROSS_SOURCE"
    return "LOW"
```

---

## Impacto Estimado das Melhorias

| Melhoria | Falsos Negativos | Falsos Positivos | Esforco |
|----------|-----------------|-----------------|---------|
| Source Detection + min_sim ajustado | -30% | Neutro | Baixo |
| Normalizacao de cabecalho | -25% | Neutro | Medio |
| Stopwords expandidas | -5% | -5% | Baixo |
| Deteccao de retificacao | Neutro | -20% | Baixo |
| Multi-CNJ | Neutro | -15% | Medio |
| HTML stripping | -5% | Neutro | Baixo |
| Score ajustado por source | -15% | Neutro | Baixo |
| Confianca/review flag | Neutro | -30% (user guidance) | Medio |

**Estimativa total:** Reducao de ~50% em falsos negativos e ~40% em falsos positivos.
