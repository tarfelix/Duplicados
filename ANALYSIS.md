# Analise Completa — Verificador de Duplicidades v5.1

## 1. Arquitetura Atual

```
app.py (413 linhas) — Orquestrador principal (Streamlit)
├── src/config.py (66 linhas) — Configuração e secrets
├── src/api/client.py (114 linhas) — HTTP client com retry para API Zion
├── src/components/ui.py (187 linhas) — Componentes visuais Streamlit
├── src/core/
│   ├── matcher.py (206 linhas) — Algoritmo de similaridade + BFS clustering
│   └── actions.py (63 linhas) — Export CSV + processamento de cancelamentos
├── src/database/
│   ├── mysql_client.py (108 linhas) — MySQL + cache Streamlit
│   ├── firestore.py (154 linhas) — Firebase init + audit log
│   └── users_firestore.py (197 linhas) — CRUD de usuarios
└── src/services/
    └── ai_explain.py (96 linhas) — OpenAI/Azure para explicação de diffs
```

Total: ~1.400 linhas de codigo ativo (excluindo legacy)

---

## 2. Bugs e Problemas Criticos

### 2.1 Cache Compartilhado entre Usuarios (GRAVE)

`mysql_client.py:52` — O `@st.cache_data` com `hash_funcs={Engine: lambda _: None}` faz com que o Engine seja ignorado no hash do cache. Os parametros `dias_historico`, `pastas`, `status` SAO parte do hash, mas se dois usuarios com filtros diferentes acessarem ao mesmo tempo, o primeiro cache domina por 30 minutos.

### 2.2 Checkbox Trigger Loop (`ui.py:172-177`)

Cada checkbox muda o state e chama `st.rerun()`, que reconstroi TODOS os grupos. Com 200 grupos x 3 itens cada = 600 checkboxes recriados por clique.

### 2.3 Set Serialization Problem (`ui.py:102-103`)

Streamlit serializa session_state como JSON internamente. `set()` vira `list` ao serializar.

### 2.4 Sem Rate Limiting no Login (`app.py:73-81`)

Nenhum controle de tentativas. Brute force possivel.

### 2.5 Senha minima de 4 caracteres

Extremamente fraco para um sistema que cancela atividades judiciais.

### 2.6 Credenciais MySQL na f-string (`mysql_client.py:25`)

Se `password` tiver caracteres especiais, a URL quebra. Deveria usar `urllib.parse.quote_plus()`.

---

## 3. Problemas de Performance

### 3.1 Algoritmo BFS O(n²) (`matcher.py:129-160`)

Para buckets de 500 itens: ~125.000 comparacoes de similaridade.

### 3.2 Rerun Completo do Script

Streamlit executa TODO o `app.py` de cima a baixo em cada interacao.
20+ chamadas a `st.rerun()` no codigo.

### 3.3 DataFrame Copiado sem Necessidade (`matcher.py:100`)

Copia todo o DataFrame antes de processar.

### 3.4 Sem Paginacao

Todos os grupos sao renderizados de uma vez.

### 3.5 `df.iterrows()` no Calculo de Metricas (`app.py:332`)

Metodo mais lento do Pandas. Deveria ser operacao vetorizada.

---

## 4. Problemas de Seguranca

| Problema | Severidade | Local |
|----------|------------|-------|
| Senha minima 4 chars | CRITICO | `users_firestore.py:87` |
| Sem rate limit no login | CRITICO | `app.py:73` |
| Sem session timeout | ALTO | Streamlit nao suporta nativamente |
| Token API sem validacao HTTPS | ALTO | `api/client.py:60` |
| Credenciais MySQL na f-string | MEDIO | `mysql_client.py:25` |
| `unsafe_allow_html=True` (6x) | MEDIO | `ui.py` |
| Firebase init race condition | BAIXO | `firestore.py:68` |
| Erro expoe schema do DB | BAIXO | `mysql_client.py:106` |

---

## 5. Avaliacao: Streamlit vs Vite + React

### Limitacoes Inerentes do Streamlit

- Rerun completo a cada interacao (2-5s por clique)
- Sem estado real no cliente
- Sem WebSocket bidirecional
- Sem routing nativo
- Sem lazy loading/virtual scroll
- Sem animacoes/transicoes
- Sem keyboard shortcuts
- Mobile quebrado
- Sem session timeout
- Sem PWA/notificacoes

### Comparativo

| Aspecto | Streamlit (Atual) | Vite + React |
|---------|-------------------|--------------|
| Responsividade UI | 2-5s por clique | <50ms |
| 500 grupos | Lento | Instant (virtual scroll) |
| Checkbox toggle | 2s delay | Instant |
| Mobile | Quebrado | Responsivo nativo |
| Autenticacao | Session state fragil | JWT + refresh token |
| Bundle size | ~15MB | ~200KB gzipped |
| Testes | Zero | Vitest + Testing Library |

### Estrategia de Migracao Recomendada (Incremental)

**Fase 1 (1 semana):** FastAPI backend reutilizando logica existente
**Fase 2 (2 semanas):** Frontend React com Vite + Tailwind + shadcn/ui
**Fase 3 (1 semana):** Polish + Deploy

---

## 6. Melhorias Imediatas (sem migracao)

### Performance
1. Paginacao de grupos
2. Substituir `df.iterrows()` por operacoes vetorizadas
3. Short-circuit no `combined_score`
4. Evitar `st.rerun()` desnecessarios
5. Remover copia desnecessaria do DataFrame

### Seguranca
1. Senha minima 12 caracteres
2. Rate limit no login
3. URL-encode credenciais MySQL
4. Nao expor erros de DB ao usuario
5. Session timeout

### Qualidade de Codigo
1. Adicionar testes
2. Substituir bare exceptions
3. Extrair magic numbers para config
4. Structured logging
5. Type hints completos
