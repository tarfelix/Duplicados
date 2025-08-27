Verificador de Duplicidade Avançado
1. Visão Geral
O Verificador de Duplicidade Avançado é uma aplicação web desenvolvida em Python com Streamlit, projetada para otimizar o fluxo de trabalho de advogados e analistas jurídicos. Sua principal função é identificar, agrupar e facilitar a gestão de publicações judiciais idênticas ou muito semelhantes que são recebidas de diferentes fontes, permitindo que o usuário cancele as duplicatas de forma rápida e segura.

A aplicação combina um algoritmo de similaridade de texto preciso com otimizações de performance para garantir uma experiência de usuário fluida, mesmo com grandes volumes de dados.

2. Principais Funcionalidades
Detecção Inteligente de Duplicatas: Utiliza um algoritmo robusto (rapidfuzz) para comparar o teor das publicações e agrupá-las com base em um score de similaridade configurável.

Performance Otimizada: Implementa "lazy loading" para carregar textos completos apenas quando necessário, reduzindo drasticamente o tempo de carregamento inicial e o consumo de memória.

Validação por Datas: Extrai e compara datas de "Publicação" e "Disponibilização" dos textos para aumentar a precisão da análise, aplicando penalidades a similaridades suspeitas e exibindo alertas visuais.

Filtros Avançados: Permite filtrar os grupos por pasta, status da atividade e oculta automaticamente grupos que já foram resolvidos (ou seja, que possuem apenas uma atividade não cancelada).

Interface Intuitiva: Facilita a seleção de uma publicação "principal" e a marcação das demais para cancelamento com poucos cliques.

Verificação Pós-Cancelamento: Após o envio dos cancelamentos para a API, a aplicação realiza uma verificação no banco de dados para confirmar a mudança de status, garantindo a integridade da operação.

Auditoria Completa: Todas as ações importantes (seleção de principal, cancelamentos, etc.) são registradas no Google Firestore para fins de auditoria, com uma interface dedicada para visualização e exportação dos logs.

Calibração de Similaridade: Uma ferramenta auxiliar que permite analisar a distribuição dos scores de similaridade por pasta, ajudando a definir os melhores limiares para cada caso.

3. Estrutura dos Arquivos
O projeto é composto por dois arquivos Python principais:

app_final_corrigido.py: O arquivo principal da aplicação Streamlit. Contém toda a lógica da interface do usuário, o fluxo de dados, a renderização dos componentes e o algoritmo de agrupamento.

api_functions_retry.py: Um módulo cliente que encapsula a comunicação com a API externa responsável por efetuar o cancelamento das atividades. Ele possui lógica de resiliência, como tentativas automáticas (retry) e limite de taxa (rate limiting).

4. Instalação e Configuração
Pré-requisitos
Python 3.9 ou superior

Passos
Clone o repositório:

git clone <url_do_seu_repositorio>
cd <pasta_do_projeto>

Crie e ative um ambiente virtual (recomendado):

python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

Instale as dependências:
O projeto utiliza as bibliotecas listadas no arquivo requirements.txt. Para instalá-las, execute:

pip install -r requirements.txt

Configure as credenciais (secrets.toml):
Para que a aplicação funcione, é necessário criar um arquivo de segredos para o Streamlit. Crie uma pasta .streamlit na raiz do projeto e, dentro dela, um arquivo chamado secrets.toml.

.streamlit/secrets.toml

# Credenciais do Banco de Dados (MySQL)
[database]
host = "SEU_HOST_DO_BANCO"
user = "SEU_USUARIO_DO_BANCO"
password = "SUA_SENHA_DO_BANCO"
name = "NOME_DO_BANCO_DE_DADOS"

# Credenciais da API de Cancelamento
[api]
url_api = "URL_BASE_DA_API"
entity_id = ID_DA_ENTIDADE # Ex: 123
token = "SEU_TOKEN_BEARER"

# Configurações do Cliente da API
[api_client]
dry_run = false # Mude para true para simular chamadas sem cancelar de fato

# Credenciais de Login na Aplicação
[credentials]
usernames = { NOME_DE_USUARIO = "SENHA_DO_USUARIO" } # Ex: tarcisio = "senha123"

# Credenciais do Google Firebase (para auditoria)
# Cole o conteúdo do seu JSON de credenciais do Firebase aqui
[firebase_credentials]
type = "service_account"
project_id = "SEU_PROJECT_ID"
private_key_id = "SUA_PRIVATE_KEY_ID"
private_key = "-----BEGIN PRIVATE KEY-----\nSUA_CHAVE_PRIVADA\n-----END PRIVATE KEY-----\n"
client_email = "SEU_CLIENT_EMAIL"
client_id = "SEU_CLIENT_ID"
auth_uri = "[https://accounts.google.com/o/oauth2/auth](https://accounts.google.com/o/oauth2/auth)"
token_uri = "[https://oauth2.googleapis.com/token](https://oauth2.googleapis.com/token)"
auth_provider_x509_cert_url = "[https://www.googleapis.com/oauth2/v1/certs](https://www.googleapis.com/oauth2/v1/certs)"
client_x509_cert_url = "URL_DO_SEU_CERTIFICADO"

5. Executando a Aplicação
Com o ambiente configurado e as dependências instaladas, inicie a aplicação com o seguinte comando no seu terminal:

streamlit run app_final_corrigido.py

A aplicação será aberta automaticamente no seu navegador padrão.
