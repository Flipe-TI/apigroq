from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
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
    model_name="llama3-8b-8192"
)

# Modelo Pydantic para a entrada JSON
class DataInput(BaseModel):
    data: List[Dict[str, Any]]
    question: str

# Função para processar a pergunta e o DataFrame
def agent_response(df: pd.DataFrame, question: str) -> str:
    agent_prompt_prefix = (
        "Responda em português e sempre retorne os valores gerados no dataframe "
        "printando na resposta e explicando os dados"
    )

    agent = create_pandas_dataframe_agent(
        llm,
        df,
        prefix=agent_prompt_prefix,
        verbose=True,
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

        # Obter a resposta da IA
        response = agent_response(df, data_input.question)

        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




