import os
import requests
import streamlit as st
from dotenv import load_dotenv
import openai  # Importação compatível com versões mais antigas
from typing import List, Dict

# Carrega variáveis de ambiente
load_dotenv()

# Configurações
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
COLLECTION_NAME = os.getenv("ASTRA_DB_COLLECTION", "holamba")
NAMESPACE = os.getenv("ASTRA_DB_NAMESPACE", "default_keyspace")
EMBEDDING_DIMENSION = 1536
ASTRA_DB_API_BASE = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configura a API da OpenAI
openai.api_key = OPENAI_API_KEY

class AstraDBClient:
    def __init__(self):
        self.base_url = f"{ASTRA_DB_API_BASE}/api/json/v1/{NAMESPACE}"
        self.headers = {
            "Content-Type": "application/json",
            "x-cassandra-token": ASTRA_DB_TOKEN,
            "Accept": "application/json"
        }
    
    def vector_search(self, collection: str, vector: List[float], limit: int = 3) -> List[Dict]:
        """Realiza busca por similaridade vetorial"""
        url = f"{self.base_url}/{collection}"
        payload = {
            "find": {
                "sort": {"$vector": vector},
                "options": {"limit": limit}
            }
        }
        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()["data"]["documents"]
        except Exception as e:
            st.error(f"Erro na busca vetorial: {str(e)}")
            st.error(f"Resposta da API: {response.text if 'response' in locals() else 'N/A'}")
            return []

def get_embedding(text: str) -> List[float]:
    """Obtém embedding do texto usando OpenAI"""
    try:
        response = openai.Embedding.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        st.error(f"Erro ao obter embedding: {str(e)}")
        return []

def generate_response(query: str, context: str) -> str:
    """Gera resposta usando o modelo de chat da OpenAI"""
    if not context:
        return "Não encontrei informações relevantes para responder sua pergunta."
    
    prompt = f"""Responda baseado no contexto abaixo:
    
    Contexto:
    {context}
    
    Pergunta: {query}
    Resposta:"""
    
    try:
        response = openai.ChatCompletion.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "Você é especialista sobre a Holambra Cooperativa Agroindustrial e em Marketing Digital que responde perguntas com base no contexto fornecido de forma atenciosa e detalhada."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Erro ao gerar resposta: {str(e)}"

def main():
    st.title("🤖 Chatbot RAG com Astra DB")
    st.write("Conectado à coleção:", COLLECTION_NAME)
    
    # Inicializa cliente do Astra DB
    astra_client = AstraDBClient()
    
    # Inicializa histórico de conversa
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Exibe mensagens anteriores
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Processa nova entrada
    if prompt := st.chat_input("Digite sua mensagem..."):
        # Adiciona mensagem do usuário ao histórico
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Obtém embedding e busca no Astra DB
        embedding = get_embedding(prompt)
        if embedding:
            results = astra_client.vector_search(COLLECTION_NAME, embedding)
            context = "\n".join([str(doc) for doc in results])
            
            # Gera resposta
            response = generate_response(prompt, context)
            
            # Adiciona resposta ao histórico
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

if __name__ == "__main__":
    main()
