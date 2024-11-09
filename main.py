from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv
import os

# Carregar variáveis de ambiente
load_dotenv(find_dotenv())
api_key = os.getenv("GROQ_API_KEY")

# Inicializando a API FastAPI
app = FastAPI()

# Inicializando o modelo de linguagem com a chave da API
llm = ChatGroq(
    temperature=0,
    api_key=api_key,
    model_name="llama-3.1-8b-instant"
)

fiscal_de_llm = ChatGroq(
    temperature=0.8, 
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-70b-8192"
)

# Modelo Pydantic para a entrada JSON
class DataInput(BaseModel):
    data: List[Dict[str, Any]]
    question: str

# Função para processar a pergunta e o DataFrame
def agent_response(llm: ChatGroq, df: pd.DataFrame, question: str) -> str:
    agent_prompt_prefix = (
        "Responda em português e sempre retorne os valores gerados no dataframe "
        "printando na resposta e explicando os dados"
    )

    agent = create_pandas_dataframe_agent(
        llm,
        df,
        prefix=agent_prompt_prefix,
        verbose=True,
        max_iterations=5,
        allow_dangerous_code=True,
        agent_executor_kwargs={"handle_parsing_errors": True}
    )

    response = agent.invoke({"input": question})
    return response.get('output')

# Endpoint para receber JSON e responder com análise da IA
@app.post("/ask")
async def ask(data_input: DataInput):
    try:
        # Criar DataFrame a partir do JSON
        df = pd.DataFrame(data_input.data)

        system = "Você é um especialista em curadoria de bot, formate a resposta a seguir da melhor forma possivel em português pensando no usuario final."
        human = "{text}"
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
        chain = prompt | fiscal_de_llm

        response = agent_response(llm,df,data_input.question)#chain.stream({"text": agent_response(llm,df,data_input.question)})
        full_response = ""
        for partial_response in response:
            full_response += str(partial_response.content)
        
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




