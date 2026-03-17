# PLANO DE IMPLEMENTACAO — Migracao Completa

## Objetivo
Migrar de Streamlit + Firestore para FastAPI + Vite/React + PostgreSQL.
Tudo neste branch (`claude/analyze-app-optimization-8bDDo`) para teste local e depois Coolify.

---

## FASE 1 — Backend FastAPI + PostgreSQL (substituir Streamlit + Firestore)

### 1.1 PostgreSQL — Schema e Models

Criar tabelas para substituir Firestore:

```sql
-- Tabela de usuarios (substitui Firestore verificador_users)
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role VARCHAR(20) NOT NULL DEFAULT 'user' CHECK (role IN ('admin', 'user')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Tabela de audit log (substitui Firestore duplicidade_actions)
CREATE TABLE audit_logs (
    id SERIAL PRIMARY KEY,
    ts TIMESTAMPTZ DEFAULT NOW(),
    username VARCHAR(100) NOT NULL,
    action VARCHAR(100) NOT NULL,
    details JSONB DEFAULT '{}'
);
```

**Arquivos a criar:**
- `backend/database/postgres.py` — Engine + sessionmaker via SQLAlchemy
- `backend/database/models.py` — User, AuditLog (ORM models)
- `backend/database/migrations/001_initial.sql` — Schema SQL

### 1.2 Backend FastAPI — Estrutura

```
backend/
├── main.py                    # FastAPI app, CORS, lifespan
├── config.py                  # Settings via pydantic-settings (env vars)
├── auth.py                    # JWT auth (login, token refresh, password hashing)
├── dependencies.py            # get_db, get_current_user, require_admin
├── database/
│   ├── postgres.py            # PostgreSQL engine + session (substituir firestore.py)
│   ├── mysql_client.py        # Reutilizar logica existente SEM st.cache
│   └── models.py              # SQLAlchemy ORM models (User, AuditLog)
├── routers/
│   ├── auth_router.py         # POST /api/auth/login, POST /api/auth/logout
│   ├── users_router.py        # CRUD /api/users (admin only)
│   ├── activities_router.py   # GET /api/activities, GET /api/activities/filters
│   ├── groups_router.py       # GET /api/groups, POST /api/groups/cancel
│   └── export_router.py       # GET /api/export/csv
├── services/
│   ├── matcher.py             # Reutilizar src/core/matcher.py (sem streamlit)
│   ├── ai_explain.py          # Reutilizar src/services/ai_explain.py (ja sem st)
│   └── api_client.py          # Reutilizar src/api/client.py (ja sem st)
└── schemas.py                 # Pydantic models para request/response
```

### 1.3 Endpoints da API

```
POST   /api/auth/login          { username, password } → { access_token, role }
POST   /api/auth/change-password { current_password, new_password }
GET    /api/auth/me              → { username, role }

GET    /api/users                (admin) → lista de usuarios
POST   /api/users                (admin) { username, password, role }
PATCH  /api/users/:username/role (admin) { role }
PATCH  /api/users/:username/password (admin) { new_password }
DELETE /api/users/:username      (admin)
GET    /api/users/has-any        → { has_users: bool }

GET    /api/activities/filters   → { pastas: [], status: [] }
GET    /api/activities?dias=10&pastas=X&status=Y → DataFrame como JSON

GET    /api/groups?dias=10&pastas=X&status=Y&min_sim=90&min_containment=55&use_cnj=true
       → { groups: [...], metrics: { total_groups, abertas, ... } }

POST   /api/groups/cancel        { items: [{ activity_id, principal_id }], dry_run }
       → { ok: N, err: N, details: [...] }

GET    /api/groups/export-csv?...filtros... → arquivo CSV
POST   /api/groups/diff          { text_a, text_b } → { html_a, html_b }
POST   /api/groups/explain-diff  { text_a, text_b } → { explanation }
```

### 1.4 O que reutilizar vs reescrever

| Modulo atual | Acao | Motivo |
|---|---|---|
| `src/core/matcher.py` | Copiar e remover `import streamlit` | Logica pura, so usa pandas/rapidfuzz |
| `src/api/client.py` | Copiar direto | Ja nao usa streamlit (so config) |
| `src/services/ai_explain.py` | Copiar direto | Ja nao usa streamlit |
| `src/database/mysql_client.py` | Reescrever sem `@st.cache_*` | Cache via `lru_cache` ou Redis |
| `src/database/firestore.py` | DELETAR | Substituido por PostgreSQL |
| `src/database/users_firestore.py` | Reescrever para PostgreSQL | Mesma logica, ORM diferente |
| `src/components/ui.py` | DELETAR | Substituido por React components |
| `src/config.py` | Reescrever | pydantic-settings ao inves de st.secrets |
| `src/core/actions.py` | Reescrever sem st.progress | Retornar resultados via response |
| `app.py` | DELETAR | Substituido por FastAPI + React |

---

## FASE 2 — Frontend React + Vite + TypeScript

### 2.1 Estrutura

```
frontend/
├── index.html
├── package.json
├── vite.config.ts
├── tsconfig.json
├── tailwind.config.ts
├── postcss.config.js
├── .env.example               # VITE_API_URL=http://localhost:8000
├── src/
│   ├── main.tsx               # Entry point
│   ├── App.tsx                # Router setup
│   ├── api/
│   │   └── client.ts          # Axios/fetch wrapper com interceptors JWT
│   ├── hooks/
│   │   ├── useAuth.ts         # Login/logout/token refresh
│   │   ├── useGroups.ts       # TanStack Query: fetch groups
│   │   ├── useActivities.ts   # TanStack Query: fetch filters + activities
│   │   └── useUsers.ts        # TanStack Query: CRUD users
│   ├── stores/
│   │   └── groupStates.ts     # Zustand: principal_id, cancelados por grupo
│   ├── pages/
│   │   ├── LoginPage.tsx      # Tela de login
│   │   ├── SetupPage.tsx      # Primeiro acesso (criar admin)
│   │   ├── DashboardPage.tsx  # Pagina principal com grupos
│   │   └── UsersPage.tsx      # Gerenciar usuarios (admin)
│   ├── components/
│   │   ├── layout/
│   │   │   ├── Sidebar.tsx    # Filtros laterais
│   │   │   ├── Header.tsx     # Barra superior
│   │   │   └── Layout.tsx     # Shell principal
│   │   ├── groups/
│   │   │   ├── GroupList.tsx        # Lista com virtual scroll
│   │   │   ├── GroupCard.tsx        # Expander de um grupo
│   │   │   ├── ActivityCard.tsx     # Card de uma atividade
│   │   │   ├── SimilarityBadge.tsx  # Badge de similaridade
│   │   │   ├── DiffDialog.tsx       # Modal de comparacao
│   │   │   └── CancelDialog.tsx     # Modal de confirmacao
│   │   ├── users/
│   │   │   ├── UserList.tsx
│   │   │   ├── UserForm.tsx
│   │   │   └── ChangePassword.tsx
│   │   └── ui/                # shadcn/ui components
│   │       ├── button.tsx
│   │       ├── dialog.tsx
│   │       ├── input.tsx
│   │       ├── badge.tsx
│   │       ├── card.tsx
│   │       ├── checkbox.tsx
│   │       ├── slider.tsx
│   │       ├── select.tsx
│   │       ├── table.tsx
│   │       ├── toast.tsx
│   │       └── ...
│   ├── lib/
│   │   └── utils.ts           # cn(), formatDate(), etc
│   └── types/
│       └── index.ts           # TypeScript interfaces
└── components.json            # shadcn/ui config
```

### 2.2 Funcionalidades do Frontend

**Login/Auth:**
- JWT armazenado em httpOnly cookie (mais seguro que localStorage)
- Refresh automático
- Redirect para login se token expirado
- Session timeout configurável

**Dashboard (pagina principal):**
- Sidebar com todos os filtros (mesmos do Streamlit)
- Filtros refletidos na URL (?dias=10&sim=90) — compartilhável
- Metricas no topo (grupos, abertas, marcados)
- Lista de grupos com TanStack Virtual (virtual scroll)
- Cada grupo: expandível, com cards de atividades
- Checkboxes instantâneos (state local via Zustand)
- Botão "Ver diferenças" abre modal com diff HTML
- Botão "Explicar com IA" dentro do modal
- "Processar Marcados" com dialog de confirmação + progress bar real
- Download CSV

**User Management (admin):**
- Tabela com usuarios, roles, ações
- Criar/editar/deletar usuarios
- Alterar senha (propria e de outros)

**UX Melhorias:**
- Dark mode toggle
- Keyboard shortcuts (Esc fechar dialogs, Enter confirmar)
- Toast notifications
- Skeleton loading
- Responsive mobile layout
- Batch select com Shift+Click

### 2.3 Dependencias Frontend

```json
{
  "dependencies": {
    "react": "^19",
    "react-dom": "^19",
    "react-router-dom": "^7",
    "@tanstack/react-query": "^5",
    "@tanstack/react-virtual": "^3",
    "zustand": "^5",
    "axios": "^1",
    "tailwindcss": "^4",
    "class-variance-authority": "^0.7",
    "clsx": "^2",
    "tailwind-merge": "^3",
    "lucide-react": "^0.400",
    "react-hook-form": "^7",
    "@hookform/resolvers": "^3",
    "zod": "^3",
    "sonner": "^1",
    "date-fns": "^4"
  },
  "devDependencies": {
    "typescript": "^5.5",
    "vite": "^6",
    "@vitejs/plugin-react": "^4",
    "vitest": "^3",
    "@testing-library/react": "^16"
  }
}
```

---

## FASE 3 — Infraestrutura e Deploy

### 3.1 Docker Compose (desenvolvimento local)

```yaml
services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: duplicados
      POSTGRES_USER: duplicados
      POSTGRES_PASSWORD: duplicados_dev
    ports: ["5432:5432"]
    volumes: ["pgdata:/var/lib/postgresql/data"]

  backend:
    build: ./backend
    ports: ["8000:8000"]
    env_file: .env
    depends_on: [postgres]
    volumes: ["./backend:/app"]  # hot-reload

  frontend:
    build: ./frontend
    ports: ["5173:5173"]
    depends_on: [backend]
    volumes: ["./frontend/src:/app/src"]  # HMR

volumes:
  pgdata:
```

### 3.2 Dockerfiles

**backend/Dockerfile:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential default-libmysqlclient-dev curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**frontend/Dockerfile (produção):**
```dockerfile
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
```

### 3.3 Coolify Deploy

Opções:
- **Opção A (recomendada):** Docker Compose no Coolify — um único serviço com 3 containers
- **Opção B:** 3 serviços separados no Coolify (mais flexível mas mais complexo)
- Nginx como reverse proxy: `/api/*` → backend:8000, `/*` → frontend static

### 3.4 Variáveis de Ambiente (backend/.env)

```env
# PostgreSQL (novo - substitui Firebase)
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=duplicados
POSTGRES_USER=duplicados
POSTGRES_PASSWORD=xxx

# MySQL (mantido - dados de atividades)
DATABASE_HOST=40.88.40.110
DATABASE_USER=tarcisio
DATABASE_PASSWORD=xxx
DATABASE_NAME=zion_flow

# API Zion (mantido)
API_URL_API=https://zflowv2api.zionbyonset.com.br/api
API_ENTITY_ID=3
API_TOKEN=xxx

# JWT
JWT_SECRET=xxx
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=480

# Similaridade (mantido)
SIMILARITY_MIN_SIM_GLOBAL=95
SIMILARITY_MIN_CONTAINMENT=55
SIMILARITY_DIFF_HARD_LIMIT=12000

# IA (opcional, mantido)
OPENAI_API_KEY=xxx
```

---

## FASE 4 — Testes

### 4.1 Backend (pytest)
- `tests/test_matcher.py` — Testar combined_score, create_groups, normalize_text
- `tests/test_auth.py` — Login, JWT, password hashing
- `tests/test_users.py` — CRUD de usuarios no PostgreSQL
- `tests/test_groups.py` — Endpoint de grupos e cancelamento
- `tests/conftest.py` — Fixtures com PostgreSQL de teste

### 4.2 Frontend (vitest)
- `src/__tests__/GroupCard.test.tsx`
- `src/__tests__/LoginPage.test.tsx`
- `src/__tests__/useGroups.test.ts`

---

## ORDEM DE EXECUCAO

1. **Criar estrutura de diretórios** (backend/, frontend/)
2. **Backend: config + database + models** (PostgreSQL + MySQL connections)
3. **Backend: auth (JWT + users no PostgreSQL)**
4. **Backend: routers (activities, groups, export)**
5. **Backend: Docker + migration SQL**
6. **Frontend: setup Vite + React + Tailwind + shadcn**
7. **Frontend: pages (Login, Setup, Dashboard, Users)**
8. **Frontend: components (GroupList, GroupCard, DiffDialog, etc)**
9. **Docker Compose completo**
10. **Testes backend**
11. **Testes frontend**
12. **Coolify config (nginx, env vars)**

Estimativa: ~15-20 arquivos backend, ~25-30 arquivos frontend, ~5 arquivos infra.
