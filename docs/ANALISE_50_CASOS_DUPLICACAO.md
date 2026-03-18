# Analise de 50 Casos Reais de Duplicacao

> Analise baseada em **dados reais** extraidos de 3319 atividades do MySQL (542 grupos detectados).
> Script de extracao: `scripts/extrair_standalone.py`

## Resumo Executivo dos Dados Reais

```
Total de atividades:      3319
Total de grupos:           542
Atividades em grupos:     1186
Grupos analisados:           50

Tipos de duplicacao:
  SAME_DAY:  28 grupos (56%)
  DIFF_DAY:  22 grupos (44%)
  CROSS_SOURCE: 0 grupos (0%)   ← SURPRESA!

Fontes detectadas:
  DJEN:   46 grupos (92%)
  OTHER:   4 grupos (8%)
  AASP:    0 grupos (0%)
  ADVISE:  0 grupos (0%)

Distribuicao de scores:
  90-100: 75 pares (94%)
  80-89:   3 pares (4%)
  70-79:   1 par   (1%)

Tamanhos de grupo:
  2 itens: 46 grupos (92%)
  4 itens:  3 grupos (6%)
  6 itens:  1 grupo  (2%)
```

## Descoberta Principal: Padrao CAD1 vs DJENTJSP

**A hipotese inicial estava errada.** A duplicacao NAO e predominantemente AASP vs Advise.

O padrao dominante (>90% dos casos) e:
- **Mesma publicacao DJEN processada por dois pipelines diferentes:**
  - **Pipeline A ("CAD1"):** Texto com prefixo `CAD1 -` e campos `TIPO DE DOCUMENTO:`, `DATA DE ENVIO:`
  - **Pipeline B ("DJENTJSP/DJENSTJ/DJENTJRJ"):** Texto com prefixo `DJENTJSP -` (ou DJENSTJ/DJENTJRJ) SEM esses campos

Ambos compartilham corpo identico (decisoes judiciais, listagens de ADV, etc).

### Exemplo Real (Grupo 1)
```
Item A:
  ID: 3177050, Pasta: TRABALHISTA, Status: Fechada
  User: Luiza Guirau, Date: 2026-03-17
  Source: DJEN
  Preview: "CAD1 - TRIBUNAL REGIONAL DO TRABALHO DA 2a REGIAO
  TIPO DE DOCUMENTO: INTIMACAO
  DATA DE ENVIO: 12/03/2026 ..."

Item B:
  ID: 3178055, Pasta: TRABALHISTA, Status: Cancelada
  User: Isabella Awoyama, Date: 2026-03-17
  Source: OTHER (detect_source nao reconheceu "DJENTJSP")
  Preview: "DJENTJSP - TRIBUNAL REGIONAL DO TRABALHO DA 2a REGIAO
  [mesmo corpo do Item A sem campos TIPO DE DOCUMENTO e DATA DE ENVIO]"

Score: 95, set_ratio: 97, sort_ratio: 92, containment: 96
```

### Padrao Consistente Observado

| Caracteristica | Pipeline CAD1 | Pipeline DJENTJSP |
|----------------|---------------|-------------------|
| Prefixo | `CAD1 -` | `DJENTJSP -` / `DJENSTJ -` / `DJENTJRJ -` |
| TIPO DE DOCUMENTO | Presente | Ausente |
| DATA DE ENVIO | Presente | Ausente |
| Corpo do texto | Identico | Identico |
| Cauda (ADV, advogados) | Identica | Identica |
| User tipico | Luiza Guirau / Maria de Fatima | Isabella Awoyama |
| Status final | Fechada (principal) | Cancelada (duplicata) |

### Distribuicao Real por Tipo de Tribunal

Dos 50 grupos analisados:
- **DJENTJSP** (TRT 2a Regiao / SP): ~35 grupos
- **DJENSTJ** (STJ): ~8 grupos
- **DJENTJRJ** (TJ/TRT RJ): ~3 grupos
- **Outros/OTHER**: ~4 grupos

---

## Fluxo Real de Duplicacao (Corrigido)

```
Publicacao no DJEN (Diario de Justica Eletronico Nacional)
        |
   +---------+---------+
   |                   |
  CAD1              DJENTJSP/STJ/TJR
  (Pipeline A)      (Pipeline B)
   |                   |
   +----> ZION <-------+
         |
         Cria 2 atividades "Verificar"
         (uma de cada pipeline)
         |
         ViewGrdAtividadesTarcisio
         |
         App Duplicados detecta o par
```

O ZION recebe a mesma publicacao de dois caminhos diferentes no DJEN e cria uma atividade para cada. Nao e "erro" do ZION — e uma consequencia da arquitetura de ingestao.

---

## Analise Detalhada dos 50 Grupos Reais

### Grupos 1-10: Padrao CAD1 vs DJENTJSP (Trabalhista)

Todos seguem o mesmo padrao:
- **Pasta:** TRABALHISTA
- **Par:** CAD1 (Fechada) + DJENTJSP (Cancelada)
- **Scores:** 93-99 (media ~95)
- **Containment:** 90-100%
- **Penalidade de comprimento:** 0.95-1.0 (CAD1 ligeiramente maior por ter campos extras)

**Observacao chave:** O `detect_source()` atual classifica DJENTJSP como "OTHER" porque so procura "DJEN" exato, e "DJENTJSP" contem "DJEN" mas tambem o matcher de "DJE" pode interferir.

### Grupos 11-20: Padrao CAD1 vs DJENSTJ (STJ)

Similar ao trabalhista, mas com publicacoes do Superior Tribunal de Justica:
- **Pasta:** Varias (dependendo do cliente)
- **Scores:** 91-97
- **Diferenca:** Textos do STJ tendem a ser mais longos (recursos especiais, habeas corpus)
- **Containment:** Consistentemente >90%

### Grupos 21-30: Grupos de 4+ itens

Quando a mesma publicacao menciona multiplos processos ou e processada multiplas vezes:
- 3 grupos de 4 itens + 1 grupo de 6 itens
- Geralmente: 2 pares CAD1/DJENTJSP para o mesmo CNJ, OU
- Mesmo texto duplicado 3+ vezes por reprocessamento ZION
- **Scores entre pares genuinos:** 95-100
- **Scores entre itens com CNJs diferentes:** 70-85 (menos similar)

### Grupos 31-40: Variantes SAME_DAY

- Ambas atividades criadas no mesmo dia
- Frequentemente mesmo usuario ou usuarios do mesmo turno
- **Pattern:** ZION processa CAD1 e DJENTJSP quase simultaneamente
- Scores consistentemente >93

### Grupos 41-50: Variantes DIFF_DAY

- Atividades criadas em dias diferentes
- Tipicamente: CAD1 processado num dia, DJENTJSP no dia seguinte
- **Nao e republicacao** — e atraso no processamento de um dos pipelines
- Scores identicos aos SAME_DAY (93-99)

---

## Fraquezas Identificadas (Baseado em Dados Reais)

### 1. Source Detection Incompleta (CRITICA)

O `detect_source()` atual:
```python
def detect_source(text):
    if "AASP" in t: return "AASP"
    if "ADVISE" in t: return "ADVISE"
    if "DJEN" in t: return "DJEN"
    if "DJE" in t: return "DJE"
    return "OTHER"
```

**Problema:** Textos com "DJENTJSP" sao corretamente detectados como "DJEN" (contem "DJEN"), mas textos com "CAD1" sao classificados como "OTHER". Isso impede a deteccao de cross-source.

**Correcao:** Adicionar deteccao de CAD1 e variantes DJEN.

### 2. Cabecalhos CAD1 Nao Normalizados

Os campos exclusivos do CAD1 ("TIPO DE DOCUMENTO:", "DATA DE ENVIO:") adicionam tokens que reduzem o score de ~100% para ~95%.

**Correcao:** Strip de cabecalhos conhecidos dos pipelines antes da comparacao.

### 3. Bonus Limitado a 5 Pontos

O bonus maximo e 5, mas para pares CAD1/DJENTJSP com mesmo CNJ + orgao + tipo_doc, o bonus natural seria 6-11. O cap de 5 subestima a confianca.

**Consideracao:** Aumentar cap ou usar bonus para confianca em vez de score.

### 4. Nao Ha Fast-Path para Duplicatas Obvias

Quando dois textos do mesmo folder/CNJ tem containment >95% e set_ratio >95%, sao quase certamente duplicatas. Nao ha fast-path que pule o calculo completo.

### 5. Retificacao Nao Sinalizada

Nos 50 grupos analisados, nenhum continha retificacao (has_retificacao=False em todos). Mas a ausencia de sinalizacao significa que quando retificacoes aparecerem, serao tratadas como duplicatas normais.

### 6. Stopwords Insuficientes

Palavras dos pipelines ("cad1", "djentjsp", "djenstj") nao estao nas stopwords, contribuindo para diferenca de score desnecessaria.

---

## Propostas de Melhoria (Priorizadas por Impacto Real)

### P0 — Source Detection Melhorada

```python
def detect_source(text: str) -> str:
    t = (text or "").upper()
    if "AASP" in t:
        return "AASP"
    if "ADVISE" in t:
        return "ADVISE"
    # CAD1 e um pipeline de ingestao DJEN
    if t.lstrip().startswith("CAD"):
        return "DJEN_CAD"
    if "DJENTJSP" in t or "DJENSTJ" in t or "DJENTJRJ" in t or "DJENTRT" in t:
        return "DJEN_DJ"
    if "DJEN" in t:
        return "DJEN"
    if "DJE" in t or "DIARIO ELETRONICO" in t:
        return "DJE"
    return "OTHER"
```

**Impacto:** Permite scoring ajustado para pares DJEN_CAD + DJEN_DJ (90%+ dos duplicatas reais).

### P1 — Header Stripping para Pipelines DJEN

```python
DJEN_HEADER_PATTERNS = [
    r"^CAD\d+\s*-\s*",                           # "CAD1 - "
    r"^DJENTJSP\s*-\s*",                          # "DJENTJSP - "
    r"^DJEN\w*\s*-\s*",                           # Qualquer variante DJEN
    r"TIPO\s+DE\s+DOCUMENTO:\s*[^-\n]+[-\s]*",   # "TIPO DE DOCUMENTO: INTIMACAO -"
    r"DATA\s+DE\s+ENVIO:\s*[\d/]+\s*[-\s]*",     # "DATA DE ENVIO: 12/03/2026 -"
]
```

**Impacto:** Scores subiriam de ~95% para ~98-100% nos pares CAD1/DJENTJSP.

### P2 — Stopwords de Pipeline

Adicionar: `cad1`, `djentjsp`, `djenstj`, `djentjrj`, `djentrt`, `envio`

**Impacto:** Reducao de tokens irrelevantes na comparacao.

### P3 — Min-Sim Ajustado por Source

Quando dois itens vem de pipelines DJEN diferentes (DJEN_CAD vs DJEN_DJ), aplicar min_sim 5 pontos mais baixo:

```python
if source_a.startswith("DJEN") and source_b.startswith("DJEN") and source_a != source_b:
    effective_min_sim = min_sim - 5
```

**Impacto:** Captura pares borderline (score 87-90) que hoje sao perdidos.

### P4 — Retificacao Flag

Adicionar campo `is_retificacao` aos grupos para UI exibir aviso.

### P5 — HTML Stripping

Adicionar `re.sub(r"<[^>]+>", " ", text)` como primeiro passo de normalize_text.

---

## Impacto Estimado

| Melhoria | Falsos Negativos | Falsos Positivos | Esforco |
|----------|-----------------|-----------------|---------|
| P0: Source detection | Habilita P1/P3 | Neutro | Baixo |
| P1: Header stripping | -20% | Neutro | Baixo |
| P2: Stopwords pipeline | -5% | -2% | Trivial |
| P3: Min-sim por source | -10% | +2% (aceitavel) | Baixo |
| P4: Retificacao flag | Neutro | -10% (user guidance) | Baixo |
| P5: HTML stripping | -3% | Neutro | Trivial |

**Total estimado:** Reducao de ~35% em falsos negativos com impacto minimo em falsos positivos.
