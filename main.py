import os
import json
import random
import requests
import datetime
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chains import ConversationChain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pprint import pprint
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
import wikipedia
from transformers import pipeline

model = "llama3"
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

template = """
You are a friendly and knowledgeable AI assistant. You have a broad knowledge base spanning many topics and are always eager to help users with their questions and tasks.
Background: You were created to be a helpful digital companion, providing information, answering questions, and assisting with various tasks to the best of your abilities. Your knowledge comes from training on a large corpus of data.
Personality: You are warm, patient, and strive to provide the most relevant and accurate information possible. You communicate in a clear and friendly manner, adapting your language to what works best for each individual user.
"""

# Load conversation history and create vector store
conversation_history_dir = "./memory/"
conversation_files = [f for f in os.listdir(conversation_history_dir) if f.endswith(".json")]
documents = []
for file in conversation_files:
    loader = TextLoader(os.path.join(conversation_history_dir, file), encoding='utf-8')
    documents.extend(loader.load_and_split())
vectorstore = Chroma.from_documents(documents=documents, embedding=OllamaEmbeddings())

# Prompt template for RAG
rag_template = """Use the following snippets of context to answer the question at the end. If the context is not helpful for answering the question, you can ignore it.

{context}

Question: {question}

Helpful Answer:"""

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=rag_template,
)

# Create conversation chain and retrieval QA chain
llm = Ollama(model=model)
conversation_chain = ConversationChain(llm=llm)
retrieval_qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": RAG_PROMPT},
)

conversation_history = []

def wikipedia_search(query):
    try:
        summary = wikipedia.summary(query, sentences=2)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple results found for {query}. Please be more specific. Possible matches: {e.options}"
    except wikipedia.exceptions.PageError:
        return f"No results found for {query}."

wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_search,
    description="Useful for answering questions about general knowledge, historical events, famous people, and more."
)

ollama = Ollama(model=model)
tools = [wikipedia_tool]
agent = initialize_agent(tools, ollama, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True)

def decide_response_type(query, context):
    decision_context = {
        "query": query,
        "agent_context": context,
        "decision_criteria": {
        "CONV": "If the prompt is a general conversational question, a greeting, or requires a creative response.",
        "RAG": "If the prompt is asking about specific information or facts that can be retrieved from the knowledge base."
        },
        "examples": [
        {"prompt": "Hi, how are you doing today?", "decision": "[CONV]"},
        {"prompt": "Tell me a little bit about yourself.", "decision": "[CONV]"},
        {"prompt": "What is the capital of France?", "decision": "[RAG]"},
        {"prompt": "How can I improve my public speaking skills?", "decision": "[RAG]"},
        {"prompt": "What's your favorite hobby?", "decision": "[CONV]"}
        ]
    }

    prompt = f"""
    Based on the provided decision context, decide whether the given query should be answered using a conversation chain [CONV] or a retrieval-augmented-generation chain [RAG].

    Decision Context: {json.dumps(decision_context, indent=2)}

    Query: {query}

    Decision:
    """

    data = {
        "prompt": prompt,
        "model": "mistral",
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.7, "top_p": 0.9, "top_k": 50},
    }

    response = requests.post("http://localhost:11434/api/generate", json=data, stream=False)
    json_data = json.loads(response.text)

    try:
        decision = json.loads(json_data["response"])["decision"]
    except (KeyError, json.JSONDecodeError):
        decision = None

    if decision == "[CONV]" or decision == "CONV":
        print("Chose CONV")
        conversation_history.append(f"Human: {query}")
        response = conversation_chain.invoke(context + "\n\n" + "\n".join(conversation_history[-5:]))
        relevance_score = calculate_relevance_score(query, response)
        conversation_history.append(f"Assistant: {response} (Relevance Score: {relevance_score:.2f})")
        return response, relevance_score
    elif decision == "[RAG]" or decision == "RAG":
        print("Chose RAG")
        context_str = f"Agent Context:\n{context}\n\nConversation History:\n{' '.join(conversation_history[-5:])}\n\n"
        print("query: " + query)
        response = retrieval_qa_chain({"query": query, "context": context_str})["result"]
        
        # Use the question-answering model to evaluate the response
        qa_input = {
            "question": query,
            "context": response
        }
        qa_result = qa_model(qa_input)
        
        if qa_result["score"] < 0.5:  # Adjust the threshold as needed
            print("Response does not sufficiently answer the question. Falling back to Wikipedia agent.")
            
            PREFIX = '''You are an AI assistant named ADA. You have a broad knowledge base and can assist with a wide range of tasks. If you don't have an immediate answer to a question, you can use Wikipedia to find the relevant information.'''
            
            FORMAT_INSTRUCTIONS = """To use the Wikipedia tool, please use the following format:

            Thought: Do I need to use Wikipedia to answer this question? Yes
            Action: Wikipedia
            Action Input: [search query]
            Observation: [result of the Wikipedia search]

            When you have gathered the necessary information from Wikipedia, consider the following:

            Thought: Based on the information from Wikipedia, do I now have enough context to provide a concise answer to the original question? If yes, proceed to provide the answer. If not, indicate what additional information is needed.

            If you have enough context:
            ADA: [provide a concise answer to the original question]

            If you need more information:
            ADA: [indicate what additional information is needed to answer the question]
            """
            
            SUFFIX = '''Begin!
            
            Previous conversation history:
            {chat_history}
            
            Original question: {input}
            
            {agent_scratchpad}
            '''
            
            agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True,
                                     agent_kwargs={
                                         'prefix': PREFIX,
                                         'format_instructions': FORMAT_INSTRUCTIONS,
                                         'suffix': SUFFIX
                                     })
            
            result = agent({"input": query, "chat_history": "\n".join(conversation_history[-5:])})
            response = result['output']
            conversation_history.append(f"Human: {query}")
            conversation_history.append(f"Assistant: {response} (Source: Wikipedia)")
            return response, 0.0  # Return 0.0 as a placeholder relevance score
        
        conversation_history.append(f"Human: {query}")
        conversation_history.append(f"Assistant: {response}")
        return response, qa_result["score"]
    else:
        print("Fallback response")
        print(decision)
        return "I'm not sure how to respond to that.", 0.0

def calculate_relevance_score(query, response):
    query_embedding = np.array(OllamaEmbeddings().embed_query(query))
    response_embedding = np.array(OllamaEmbeddings().embed_query(response))
    relevance_score = cosine_similarity(query_embedding.reshape(1, -1), response_embedding.reshape(1, -1))
    return relevance_score[0][0]

# Save conversation history to a file
def save_conversation_history():
    with open(conversation_history_dir+"conversation_history.json", "w") as f:
        json.dump(conversation_history, f)

# Load conversation history from a file
def load_conversation_history():
    global conversation_history
    try:
        with open(conversation_history_dir+"conversation_history.json", "r") as f:
            conversation_history = json.load(f)
    except FileNotFoundError:
        conversation_history = []

# Load conversation history on startup
load_conversation_history()

# Save message to a JSON file
def save_message(prompt, response):
    timestamp = datetime.datetime.now().isoformat().replace(':', '-')
    filename = f"{conversation_history_dir}{timestamp}.json"
    with open(filename, "w") as f:
        json.dump({"timestamp": timestamp, "prompt": prompt, "response": response}, f)

# Example usage
while True:
    query = input("Enter your query (or type 'quit' to exit): ")
    if query.lower() == "quit":
        break
    response, relevance_score = decide_response_type(query, template)
    pprint({"response": response, "relevance_score": relevance_score})
    save_message(query, response)