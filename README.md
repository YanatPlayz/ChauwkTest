
# Chauwk Assistance Chatbot


This repository holds custom code for a chatbot that answers questions about local pdf data.

It stores the document embeddings in a persistent Chroma database. The chatbot is based on OpenAI's APIs, using the  *GPT-3.5-turbo-0125* model for the conversational chat and the *Text-embedding-ada-002-v2* model for embedding tasks.

Audio support in development, where users can converse with the bot in Indian languages with the help of Bhashini API.

*Tanay Agrawal*

## API Reference
This project uses OpenAI's APIs. However, integrating HuggingFace models instead is a entirely free option, albeit slower than OpenAI.

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `OPENAI_API_KEY` | `string` | Your OpenAI API key. **Required**  |
| `HUGGINGFACEHUB_API_TOKEN` | `string` | If needed. **Optional**  |
| `userId` | `string` | Bhashini API userId. **Required**  |
| `ulcaApiKey` | `string` | Bhashini ulcaApiKey. **Required**  |
| `InferenceApiKey` | `string` | Bhashini InferenceApiKey. **Required**  |

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

