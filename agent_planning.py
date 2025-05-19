
###
#  Autor : Jairo Monassa
import os
import json
from dotenv import load_dotenv
import openai
# import azure.identity # Removido pois não está sendo usado

load_dotenv(override=True)
OUTPUT_FOLDER = "study_plans"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Inicializa gitmodel
client = None
MODEL_NAME = None

api_key = os.getenv("GITHUB_TOKEN")
if not api_key:
    raise ValueError("Variável de ambiente GITHUB_TOKEN não definida.")
client = openai.OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=api_key,
)
MODEL_NAME = os.getenv("GITHUB_MODEL", "gpt-4o")
print(f"Usando GitHub Models com modelo: {MODEL_NAME}")

# --- Definição da Ferramenta (Função Python) --- (Sem alterações)
def save_study_plan_to_json(study_plan: dict) -> str:
    """
    Salva um plano de estudos estruturado (dicionário) em um arquivo JSON
    com nome fixo 'plano.json'.

    Args:
        study_plan: O dicionário Python contendo o plano de estudos estruturado.

    Returns:
        Uma string indicando sucesso ou falha.
    """
    try:
        filename = "planning2.json"
        filepath = os.path.join(OUTPUT_FOLDER, filename)
        print(f"dict: {study_plan}")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(study_plan, f, indent=2, ensure_ascii=False)
        print(f"\n[INFO] Plano de estudos salvo com sucesso em: {filepath}")
        return f"Plano de estudos salvo com sucesso em {filepath}"
    except Exception as e:
        print(f"\n[ERRO] Erro ao salvar o arquivo JSON: {e}")
        return f"Erro ao salvar o arquivo: {e}"

# --- Descrição da Ferramenta para a API OpenAI --- (Sem alterações)
tools = [
    {
        "type": "function",
        "function": {
            "name": "save_study_plan_to_json",
            "description": "Salva o plano de estudos estruturado gerado em um arquivo JSON local chamado 'plano.json'. Deve ser chamada APENAS DEPOIS de coletar as informações do usuário e gerar o plano.",
            "parameters": {
                "type": "object",
                "properties": {
                    "study_plan": {
                        "type": "object",
                        "description": (
                            "O objeto JSON contendo o plano de estudos estruturado. "
                            "Deve seguir o formato com chaves como 'semana1', 'semana2', etc. "
                            "Dentro de cada semana, chaves como 'dias1e2', 'dia3', etc. "
                            "E dentro de cada bloco de dias, as chaves 'topico' (string), "
                            "'subtopicos' (array de strings), e 'meta' (string)."
                        ),
                    },
                },
                "required": ["study_plan"],
            },
        },
    }
]

# --- Função Principal --- MODIFICADA (Loop de Conversa e System Prompt) ---
def main():
    subject = input("Olá! Sou seu assistente de planejamento de estudos. Qual assunto você gostaria de planejar? ")

    messages = [
        {
            "role": "system",
            # --- PROMPT DO SISTEMA MODIFICADO ---
            "content": (
                "Você é um assistente especialista em planejamento de estudos. Seu objetivo é criar um plano de estudos personalizado e estruturado. "
                f"O assunto principal é '{subject}'.\n"
                "**ETAPA 1: Coleta de Informações**\n"
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
            ),
        },
        {
            "role": "user",
            # Mensagem inicial do usuário já foi coletada no input()
            "content": f"Gostaria de um plano de estudos para {subject}."
        },
    ]

    print(f"\nOk, vamos planejar seus estudos para {subject}.")
    plan_saved = False # Flag para controlar se o plano já foi salvo

    while True:
        try:
            # --- Chamada à API OpenAI ---
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=tools,
                tool_choice="auto", # Deixa o modelo decidir quando chamar a ferramenta
            )

            response_message = response.choices[0].message
            messages.append(response_message) # Adiciona a resposta do assistente ao histórico

            # --- Processamento da Resposta ---

            # 1. Verificar se o modelo quer chamar a ferramenta
            tool_calls = response_message.tool_calls
            if tool_calls:
                print("\n[INFO] O assistente está tentando salvar o plano...")
                available_functions = {
                    "save_study_plan_to_json": save_study_plan_to_json,
                }

                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = available_functions.get(function_name)

                    if function_name == "save_study_plan_to_json" and function_to_call:
                        try:
                            print(f"[DEBUG] Arguments : {tool_call.function.arguments}")
                            function_args = json.loads(tool_call.function.arguments)
                      
                            print(f"[DEBUG] Parsed arguments dict: {function_args}")

                            # Chama a função Python real
                            function_response = function_to_call(
                                study_plan=function_args   ## [DEBUG] passar todo o argumento, pois não tem a chave .get("study_plan")
                            )

                            # Adiciona o resultado da ferramenta ao histórico
                            messages.append(
                                {
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": function_name,
                                    "content": function_response,
                                }
                            )
                            plan_saved = True # Marca que o plano foi salvo

                        except json.JSONDecodeError:
                            error_msg = f"Erro: Argumentos inválidos (não JSON) para {function_name}."
                            print(f"\n[ERRO] {error_msg}")
                            messages.append({
                                "tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": error_msg,
                            })
                        except Exception as e:
                             error_msg = f"Erro ao executar a função {function_name}: {e}"
                             print(f"\n[ERRO] {error_msg}")
                             messages.append({
                                 "tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": error_msg,
                             })
                    else:
                        # Função desconhecida ou não é a de salvar
                        print(f"\n[AVISO] Modelo tentou chamar função desconhecida: {function_name}")
                        messages.append({
                            "tool_call_id": tool_call.id, "role": "tool", "name": function_name,
                            "content": f"Erro: Função '{function_name}' não encontrada.",
                        })


                if plan_saved:
                    print("[INFO] Obtendo confirmação final do assistente...")
                    final_response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages # Envia histórico incluindo resultado da ferramenta
                    )
                    final_message = final_response.choices[0].message
                    messages.append(final_message) # Adiciona confirmação final ao histórico
                    print(f"\nAssistente: {final_message.content}")
                    print("\n[INFO] Processo concluído.")
                    break # Termina o loop principal

            # 2. Se não chamou ferramenta, apenas exibe a mensagem (provavelmente uma pergunta)
            else:
                assistant_message_content = response_message.content
                if assistant_message_content:
                    print(f"\nAssistente: {assistant_message_content}")
                else:
                    # Pode acontecer se o modelo só decidiu chamar a ferramenta mas falhou antes
                    print("\n[INFO] O assistente não forneceu uma resposta textual.")
                    # Poderia adicionar uma lógica para tentar novamente ou sair

                # --- Coleta a próxima entrada do usuário ---
                user_input = input("Você: ").strip()
                if user_input.lower() == "sair":
                    print("\nEncerrando a conversa.")
                    break
                if not user_input:
                    continue

                messages.append({"role": "user", "content": user_input})

        except openai.APIError as e:
            print(f"\n[ERRO] Erro na API OpenAI: {e}")
            break
        except openai.AuthenticationError as e:
             print(f"\n[ERRO] Erro de Autenticação OpenAI: {e}")
             print("Verifique sua API Key ou Token.")
             break
        except Exception as e:
            print(f"\n[ERRO] Ocorreu um erro inesperado: {e}")
            break

if __name__ == "__main__":
    main()
