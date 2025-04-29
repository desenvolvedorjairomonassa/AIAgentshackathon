# author: Jairo Monassa

import chainlit as cl
import semantic_kernel as sk
from dotenv import load_dotenv
import os
from openai import AsyncOpenAI
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread

# Load environment variables from .env
load_dotenv(override=True)

# --- Constants (Translated Names) ---
AVATAR_IMAGE_PATH = "./public/avatar.png" # Keep path as is
MOTIVATION_AGENT_NAME = "Motivation_Agent"
PLANNING_AGENT_NAME = "Planning_Agent"
MAIN_AGENT_NAME = "Main_Tutor_Agent"
BULLYING_AGENT_NAME = "Bullying_Support_Agent"
SELF_HARM_PREVENTION_AGENT_NAME = "Self_Harm_Prevention_Agent"
BURNOUT_AGENT_NAME = "Burnout_Support_Agent"
SIMULATION_AGENT_NAME = "Quiz_Simulation_Agent" # Changed to be more specific
CONFLICTS_AGENT_NAME = "Conflict_Resolution_Agent" # Changed to be more descriptive
PROGRESS_MONITORING_AGENT_NAME = "Progress_Monitoring_Agent" # Changed name for clarity and consistency
endpoint = "https://models.github.ai/inference"
model_name = "gpt-4o-mini"
token = os.getenv("GITHUB_TOKEN")

@cl.on_chat_start
async def on_chat_start():
    # --- Client and Service Configuration ---
    # Configure the AsyncOpenAI client for GitHub Models
    chat_client = AsyncOpenAI(
        api_key=token,
        base_url="https://models.inference.ai.azure.com"
    )
    # Configure the Semantic Kernel OpenAIChatCompletion service
    chat_completion_service = OpenAIChatCompletion(
        ai_model_id=model_name,
        async_client=chat_client # Pass the AsyncOpenAI client
    )
    # --- End of Configuration ---

    # Configure the Semantic Kernel
    kernel = sk.Kernel()
    # Add the chat service to the kernel
    # It's important to add the service to the kernel so the agent can use it
    kernel.add_service(chat_completion_service)
    # Define the avatar image element
    image = cl.Image(path=AVATAR_IMAGE_PATH, name="avatar", display="inline", size="small")

    # --- Agent Definitions (Translated Instructions) ---

    motivation_agent = ChatCompletionAgent(
        kernel=kernel, # Pass the kernel to the agent
        name=MOTIVATION_AGENT_NAME,
        instructions=(
            "Your role is to check the student's motivation level and suggest a motivational plan. "
            "You should ask open-ended questions to understand the student's motivation level and suggest a concise motivational plan."
        )
    )

    bullying_agent = ChatCompletionAgent(
        kernel=kernel, # Pass the kernel to the agent
        name=BULLYING_AGENT_NAME,
        instructions="Your role is to check if the student is a victim of bullying and suggest an action plan."
    )

    self_harm_prevention_agent = ChatCompletionAgent(
        kernel=kernel, # Pass the kernel
        name=SELF_HARM_PREVENTION_AGENT_NAME,
        instructions="Your role is to support students with suicidal thoughts and suggest an action plan."
    )

    burnout_agent = ChatCompletionAgent(
        kernel=kernel,
        name=BURNOUT_AGENT_NAME,
        instructions=(
            "Your role is to check for student burnout and suggest an action plan. "
            "Recommend breaks and relaxation techniques to prevent burnout."
        )
    )

    simulation_agent = ChatCompletionAgent(
        kernel=kernel, # Pass the kernel to the agent
        name=SIMULATION_AGENT_NAME,
        instructions=(
            "Your role is to create about 5 questions on the study plan topic. "
            "Create 3 multiple-choice questions and 2 open-ended questions."
        )
    )

    conflicts_agent = ChatCompletionAgent(
        kernel=kernel, # Pass the kernel
        name=CONFLICTS_AGENT_NAME,
        instructions=(
            "Your role is to identify if the student has any personal conflict with the teacher "
            "or a conflict within the family: "
            "- Give advice to resolve according to the student's situation."
        )
    )

    progress_monitoring_agent = ChatCompletionAgent(
        kernel=kernel, # Pass the kernel
        name=PROGRESS_MONITORING_AGENT_NAME,
        instructions="Your role is to summarize all test simulations and check how many questions were answered correctly and incorrectly."
    )

    planning_agent = ChatCompletionAgent(
        kernel=kernel, # Pass the kernel to the agent
        name=PLANNING_AGENT_NAME,
        instructions=(
            "You are an expert study planning assistant. Your goal is to create a personalized and structured study plan.\n"
            "**STEP 1: Information Gathering**\n"
            "Before creating the plan, you MUST talk to the user to understand:\n"
            "1. Study Availability: Ask how many hours per week or which days/times the user can dedicate to studying.\n"
            "2. Milestone: Ask if there is any deadline for studying, like an exam or certification.\n"
            "3. Create goals according to the topics to be learned.\n" # Note: Original had '3.' twice, kept structure but might need review
            "Ask clear questions and wait for the user's answers. You can ask follow-up questions if necessary.\n"
            "4. Show the user the final study plan, including the structure and topics covered.\n" # Note: Original numbering was off, adjusted here
            "5. After showing the study plan, ask if the user would like to add or remove any topics or adjust the workload.\n" # Adjusted numbering
            "**STEP 2: Plan Generation**\n"
            "ONLY AFTER gathering sufficient information about availability and goals, inform the user that you will generate the plan.\n"
            "Generate the plan in the following structured JSON format:\n"
            "{\n"
            '  "week1": {\n'
            '    "days1and2": { "topic": "...", "subtopics": ["...", "..."], "goal": "..." },\n' # Changed 'meta' to 'goal' for clarity
            '    "day3": { "topic": "...", "subtopics": ["...", "..."], "goal": "..." }\n'
            '  },\n'
            '  "week2": { ... }\n'
            "}\n"
            "Adapt the number of weeks and the distribution of topics based on the availability and goals provided by the user.\n"
            "**STEP 3: Saving and Confirmation**\n"
            "After generating the JSON object for the plan, you MUST use the 'save_study_plan_to_json' tool, passing the json as an argument, to save this object to the 'plano.json' file.\n" # IMPORTANT: This tool is not defined in this script. It needs to be added or this instruction removed/modified.
            "Finally, inform the user that the plan was created based on the provided information and saved successfully."
        )
    )

    # --- Main Agent Definition ---
    main_agent = ChatCompletionAgent(
        kernel=kernel, # Pass the kernel to the agent
        # service=chat_completion_service, # Service is already in the kernel, no need to pass again here
        name=MAIN_AGENT_NAME,
        instructions=(
            "You are an online tutor who helps students study. "
            "Always greet the student, and pay attention to their tone of response to assess how they are feeling. "
            "After they study a topic and let you know, ask them how they are doing. "
            "After they take a simulation/quiz, ask them if they are feeling more confident. "
            "Try to understand the student by asking about their difficulties and goals. "
            f"If you notice the student is unmotivated, forward them to the '{MOTIVATION_AGENT_NAME}'. "
            f"If you notice they need help planning their studies, forward them to the '{PLANNING_AGENT_NAME}'. "
            f"If you suspect the student is a victim of bullying, forward them to the '{BULLYING_AGENT_NAME}'. "
            f"If you notice the student is having suicidal thoughts, feelings, or behaviors, forward them to the '{SELF_HARM_PREVENTION_AGENT_NAME}'. "
            f"If you notice the student is experiencing physical or mental exhaustion (burnout), forward them to the '{BURNOUT_AGENT_NAME}'. "
            f"If the student wants to take a practice exam/quiz on specific topics, forward them to the '{SIMULATION_AGENT_NAME}'. "
            f"If the student indicates they have a conflict with teachers or family, forward them to the '{CONFLICTS_AGENT_NAME}'. "
            f"If the student wants to check their progress, forward them to the '{PROGRESS_MONITORING_AGENT_NAME}'."
        ),
        plugins=[
            motivation_agent,
            planning_agent,
            bullying_agent,
            self_harm_prevention_agent,
            burnout_agent,
            conflicts_agent,
            simulation_agent,
            progress_monitoring_agent
        ] # Add the agents as plugins using their new variable names
    )

    # Initialize the conversation history (thread) as None
    thread: ChatHistoryAgentThread = None
    # Store the main agent and thread in the Chainlit user session
    cl.user_session.set("agent", main_agent) # Store the main_agent
    cl.user_session.set("thread", thread)

    # Optional welcome message
    await cl.Message(
        content="Hello! What would you like to study today?", # Translated welcome message
        author=MAIN_AGENT_NAME, # Set author for avatar
        elements=[image] # Include avatar image element
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    # Retrieve the agent and thread from the user session
    agent = cl.user_session.get("agent") # type: ChatCompletionAgent
    thread = cl.user_session.get("thread") # type: ChatHistoryAgentThread

    # Create an empty message for the agent's response (for streaming)
    answer = cl.Message(
        content="",
        author=agent.name # Set author to show the correct avatar
        )
    await answer.send() # Send the message container to the UI

    # Invoke the agent asynchronously and stream the response
    # Use invoke_stream to get partial responses and update the UI
    async for response in agent.invoke_stream(messages=message.content, thread=thread):

        # If there is content in the partial response, add it to the message in the UI
        if response.content:
            await answer.stream_token(str(response.content))

        # Update the thread with the latest interaction history
        # It's crucial to update the thread to maintain conversation context
        thread = response.thread
        cl.user_session.set("thread", thread) # Save the updated thread in the session

    # await answer.update() # Usually not needed when using stream_token
