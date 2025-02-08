
# Chauwk Assistance Chatbot

This is a RAG system to assist blue-collar workers in developing countries who are searching for job positions and relevant government schemes. This folder holds custom code for a chatbot that answers questions in both text and speech input about local PDF data. Users can converse with the bot in Indian languages with the help of Bhashini API.

*Tanay Agrawal*

## API Reference
This project uses the 'llama3.3-70b' model for LLM tasks through Cerebras API and LlamaParse + img2table for document parsing. However, integrating HuggingFace and/or other models instead is a entirely free option. Add these to a .env file to load using dotenv.

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `OPENAI_API_KEY` | `string` | Your OpenAI API key. **Optional**  |
| `LLAMA_CLOUD_API_KEY` | `string` | LlamaCloud API key for parsing. **Required**|
| `userId` | `string` | Bhashini API userId. **Required**  |
| `ulcaApiKey` | `string` | Bhashini ulcaApiKey. **Required**  |
| `InferenceApiKey` | `string` | Bhashini InferenceApiKey. **Required**  |
| `HUGGINGFACEHUB_API_TOKEN` | `string` | If needed. **Optional**  |
| `CEREBRAS_API_KEY` | `string` | **Required** |
| `COHERE_API_KEY` | `string` | **Required** |

## Run Locally

**Clone the project**

```bash
git clone https://github.com/YanatPlayz/ChauwkTest/
```

**Go to the project directory**

```bash
cd ChauwkTest
```

**Install dependencies**

*Creating a virtual environment for the project is recommended.*

```bash
pip install -r requirements.txt
```

**Set up database**

*First, add PDF data to the **/data** folder.*

```bash
python populate_database.py
```

**Start the streamlit app**

```bash
streamlit run app.py
```

## Appendix

Integrated bhashini-translator module from https://github.com/dteklavya/bhashini_translator, a big thank you to whoever created this!
