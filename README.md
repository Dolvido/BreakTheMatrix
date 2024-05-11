# EscapeTheMatrix

This project is an AI conversational assistant that utilizes a combination of conversation chains, retrieval-augmented generation (RAG) chains, and Wikipedia searches to provide relevant and engaging responses to user queries.

## Features

- Employs a decision-making process to determine the most appropriate response type (conversation chain or RAG chain) based on the user's query.
- Utilizes a conversation chain for general conversational questions, greetings, and creative responses.
- Leverages a RAG chain for answering specific information or facts retrieved from the knowledge base.
- Falls back to Wikipedia searches using an agent-based approach when the RAG chain response doesn't sufficiently answer the question.
- Calculates a relevance score between the user's query and the assistant's response using cosine similarity of embeddings.
- Saves conversation history to JSON files for later retrieval and analysis.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ai-conversational-assistant.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up the necessary environment variables and configurations.

## Usage

1. Run the main script:
   ```
   python main.py
   ```

2. Enter your query when prompted. The assistant will process your query and provide a relevant response.

3. To exit the program, type `'quit'` when prompted for a query.

## Customization

- Modify the `template` variable to customize the assistant's background, personality, and other characteristics.
- Adjust the decision criteria and examples in the `decide_response_type` function to fine-tune the response type selection process.
- Customize the `RAG_PROMPT` template to alter the format of the retrieval-augmented generation prompt.
- Modify the `wikipedia_search` function to handle different types of Wikipedia search results and exceptions.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- This project utilizes the [LangChain](https://github.com/hwchase17/langchain) library for building the conversational AI assistant.
- The embeddings and vectorstore are powered by [Ollama](https://github.com/OllaMac/olamollm) and [Chroma](https://github.com/chroma-core/chroma).
- The question-answering model used for evaluating responses is [DistilBERT](https://huggingface.co/distilbert-base-cased-distilled-squad) from Hugging Face.

