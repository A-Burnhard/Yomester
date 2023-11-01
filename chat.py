from typing import Final
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import os
import certifi
import logging
import json
import requests
import regex
import datetime


#BACKEND- LANGCHAIN
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.globals import set_llm_cache, get_llm_cache 
from langchain.vectorstores import Chroma


# Get the LLM cache
llm_cache = get_llm_cache()


#Ideally use virtual environment
openai_api_key = ''

#openai_api_key = os.environ.get('OPENAI_API_KEY')
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)



current_date = datetime.datetime.now().date()
if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"
print(llm_name)


loader = PyPDFLoader("data.pdf")
pages = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(pages)

#Define embeddings
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

# create vector database from data
db = DocArrayInMemorySearch.from_documents(documents, embedding)

k=10

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

#memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#Prompt template
context = ""
template = """ Use this  context to reply in three sentences or less:{context}.Question: {question} """

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0301", openai_api_key=openai_api_key, temperature=0.9)

# Run chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)


#Telegram Interface
TOKEN: Final = '6612204628:AAHH0Yms01L8GCQKelcTcKe4hqnPGtTMvhA'
BOT_USERNAME : Final = '@yomesther_bot'

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("....Text Here...... ")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("....Text Here...... ")




async def handle_message(update, context):
    text = str(update.message.text).lower()
    question = text

    username = update.message.from_user.username
    print(f'User ({username}) "{text}"')

    response = qa_chain({"query": question})
    response = response['result']
    
    if response:
        await update.message.reply_text(response)
        print(f'Bot: "{response}"')
    else:
        await update.message.reply_text("I don't understand your question.")



if __name__ == '__main__':
    print('Starting bot....')
    app = Application.builder().token(TOKEN).build()

    # commands
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', help_command))

    # Messages
    app.add_handler(MessageHandler(filters.TEXT, handle_message))

    # Errors

   # app.add_error_handler(error)

    # polls the bot
    print('Polling...')
    app.run_polling(poll_interval=2)



