# autor : Jairo Monassa

import chainlit as cl
import semantic_kernel as sk
#from dotenv import load_dotenv
import os
from openai import AsyncOpenAI 
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread 


# Carrega variáveis de ambiente do .env
#load_dotenv(override=True)

AVATAR_IMAGE_PATH = "./public/avatar.png" 
MOTIVATION_AGENT_NAME = "Agente_Motivacao"
PLANNING_AGENT_NAME = "Agente_Planejamento"
MAIN_AGENT_NAME = "Agente_Principal"
BULLYING_AGENT_NAME = "Agente_Bullying"
COMMIT_SUICIDE_AGENT_NAME = "Agente_Suicidio"
BURNOUT_AGENT_NAME = "Agente_Burnout"

@cl.on_chat_start
async def on_chat_start():
    # --- Configuração do Cliente e Serviço (movido de semantickernel_basic.py) ---
    # Configura o cliente AsyncOpenAI para GitHub Models
    chat_client = AsyncOpenAI(
        #api_key=os.environ["GITHUB_TOKEN"],
        api_key=os.environ.get("GITHUB_TOKEN"),
        base_url="https://models.inference.ai.azure.com"
    )
    # Configura o serviço Semantic Kernel OpenAIChatCompletion
    chat_completion_service = OpenAIChatCompletion(
        ai_model_id=os.getenv("GITHUB_MODEL", "gpt-4o"),
        async_client=chat_client # Passa o cliente AsyncOpenAI
    )
    # --- Fim da Configuração ---

    # Configura o Kernel do Semantic Kernel
    kernel = sk.Kernel()
    # Adiciona o serviço de chat ao kernel
    # É importante adicionar o serviço ao kernel para que o agente possa usá-lo
    kernel.add_service(chat_completion_service)
    image = cl.Image(path=AVATAR_IMAGE_PATH, name="avatar", display="inline", size="small")

    agentMotivacao = ChatCompletionAgent(
        kernel=kernel, # Passa o kernel para o agente
        name=MOTIVATION_AGENT_NAME,
        instructions=("seu papel é verificar o nível de motivação do aluno e sugerir um plano motivacional. "
                     "Você deve fazer perguntas abertas para entender o nível de motivação do aluno e sugerir um plano motivacional. "
                     )
    )
    agentBullying = ChatCompletionAgent(
        kernel=kernel, # Passa o kernel para o agente
        name=BULLYING_AGENT_NAME,
        instructions="seu papel é verificar se o aluno está sendo vítima de bullying e sugerir um plano de ação. "
    )
    agentSuicidio = ChatCompletionAgent(
        kernel=kernel, # Passa o kernel
        name=COMMIT_SUICIDE_AGENT_NAME,
        instructions="seu papel com pensamentos suicidas e sugerir um plano de ação." 
    )
    agentBurnout = ChatCompletionAgent(
        kernel=kernel, 
        name=BURNOUT_AGENT_NAME,
        instructions=("seu papel e verifica o escotamento do aluno e sugerir um plano de ação."
                    "Recomendações de pausas e técnicas de relaxamento para evitar burnout.")
    )
    agentPlanejamento = ChatCompletionAgent(
        kernel=kernel, # Passa o kernel para o agente
        name=PLANNING_AGENT_NAME,
        instructions="Você é um assistente especialista em planejamento de estudos. Seu objetivo é criar um plano de estudos personalizado e estruturado. "
                f"**ETAPA 1: Coleta de Informações**\n"
                "Antes de criar o plano, você DEVE conversar com o usuário para entender:\n"
                "1. Disponibilidade de estudo: Pergunte quantas horas por semana ou quais dias/horários o usuário pode dedicar aos estudos.\n"
                "2. Milestone: Pergunte se tem algum prazo para estudo como um exame ou certificado.\n"
                "3. Crie metas de acordo com os tópicos que vai apreender "
                "Faça perguntas claras e aguarde as respostas do usuário. Você pode fazer perguntas de acompanhamento se necessário.\n"
                "3. Mostre ao usuário o resultado final do plano de estudos, incluindo a estrutura e os tópicos abordados.\n"
                "4. Depois de mostrar o plano de estudos, pergunte se o usuário gostaria de adicionar ou remover algum tópico ou ajustar a carga horária.\n"
                "**ETAPA 2: Geração do Plano**\n"
                "SOMENTE APÓS coletar informações suficientes sobre disponibilidade e metas, informe ao usuário que você irá gerar o plano.\n"
                "Gere o plano no seguinte formato JSON estruturado:\n"
                "{\n"
                '  "semana1": {\n'
                '    "dias1e2": { "topico": "...", "subtopicos": ["...", "..."], "meta": "..." },\n'
                '    "dia3": { "topico": "...", "subtopicos": ["...", "..."], "meta": "..." }\n'
                '  },\n'
                '  "semana2": { ... }\n'
                "}\n"
                "Adapte a quantidade de semanas e a distribuição de tópicos com base na disponibilidade e metas informadas pelo usuário.\n"
                "**ETAPA 3: Salvamento e Confirmação**\n"
                "Após gerar o objeto JSON do plano, você DEVE OBRIGATORIAMENTE usar a ferramenta 'save_study_plan_to_json' passando o json como argumento para salvar este objeto no arquivo 'plano.json'.\n"
                "Finalmente, informe ao usuário que o plano foi criado com base nas informações fornecidas e salvo com sucesso."
        # O 'service' não é mais passado diretamente aqui, pois está no kernel
    )
    # Instancia o agente usando o kernel (em vez de passar o 'service' diretamente)
    # O agente usará o serviço adicionado ao kernel
    # o agente de simulador quizzes usando edubase -  EduBase - Interact with EduBase, a comprehensive e-learning platform with advanced quizzing, exam management, and content organization capabilities
    # agente para agendamento dos quizzes
    # o chat deveria pergunta intrisecamente como o aluno esta sem levantat suspeita (suti) e não esqueça de falar como está se sentido
    # conflito com os professores outros alunos   
    # conflito com pais/familiares
    # acompanha a evolução
    agent = ChatCompletionAgent(
        kernel=kernel, # Passa o kernel para o agente
        service=chat_completion_service, # Passa o serviço de chat ao agente
        name=MAIN_AGENT_NAME,
        instructions=("Você é um tutor online que ajuda os alunos a estudar. "
                      "Procure entender o aluno, perguntando sobre suas dificuldades e objetivos. "
                      f"Se você perceber que está desmotivado, encaminhe-o para o '{MOTIVATION_AGENT_NAME}'. "
                      f"Se você perceber que precisa de ajuda para planejar os estudos, encaminhe-o para o '{PLANNING_AGENT_NAME}'. "
                      f"Se você perceber que está sendo vítima de bullying, encaminhe-o para o '{BULLYING_AGENT_NAME}'. "
                      f"Se você perceber que o aluno está com pensamentos, sentimentos ou comportamentos suicidas, encaminhe-o para  '{COMMIT_SUICIDE_AGENT_NAME}'. "
                      f"Se você perceber que o aluno está com esgotamento físico ou mental, encaminhe-o para o '{BURNOUT_AGENT_NAME}'."
                      ),
        plugins=[agentMotivacao, agentPlanejamento, agentBullying, agentSuicidio, agentBurnout] # Adiciona os agentes como plugins
    )

    # Inicializa o histórico da conversa (thread) como None
    thread: ChatHistoryAgentThread = None
    # Armazena o agente e o thread na sessão do usuário do Chainlit
    cl.user_session.set("agent", agent)
    cl.user_session.set("thread", thread)

    # Mensagem opcional de boas-vindas
    await cl.Message(content="Tudo bem, vamos estudar o que hoje?",
                     elements=[image]).send()
# @cl.step(type="tool")
# async def tool():
#     # Fake tool
#     await cl.sleep(2)
#     return "Response from the tool!"

@cl.on_message
async def on_message(message: cl.Message):
    # Recupera o agente e o thread da sessão do usuário
    agent = cl.user_session.get("agent") # type: ChatCompletionAgent
    thread = cl.user_session.get("thread") # type: ChatHistoryAgentThread
    # Call the tool
    #tool_res = await tool()

    # Cria uma mensagem vazia para a resposta do agente (para streaming)
    answer = cl.Message(content="")
    await answer.send() # Envia o contêiner da mensagem para a UI

    # Invoca o agente de forma assíncrona e faz streaming da resposta
    # Use invoke_stream para obter respostas parciais e atualizar a UI
    async for response in agent.invoke_stream(messages=message.content, thread=thread):

        # Se houver conteúdo na resposta parcial, adiciona-o à mensagem na UI
        if response.content:
            await answer.stream_token(str(response.content))

        # Atualiza o thread com o histórico mais recente da interação
        # É crucial atualizar o thread para manter o contexto da conversa
        thread = response.thread
        cl.user_session.set("thread", thread) # Salva o thread atualizado na sessão
