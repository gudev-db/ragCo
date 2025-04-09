import os
import requests
import streamlit as st
from dotenv import load_dotenv
import openai
from typing import List, Dict
from PIL import Image
import io

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

# URL da imagem do ícone do bot (substitua pelo seu link)
BOT_ICON_URL = "ss.png"  # Substitua pelo seu link real

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

def load_bot_icon():
    """Carrega a imagem do ícone do bot"""
    try:
        response = requests.get(BOT_ICON_URL)
        img = Image.open(io.BytesIO(response.content))
        return img
    except Exception as e:
        st.warning(f"Não foi possível carregar o ícone do bot: {str(e)}")
        return None

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
                {"role": "system", "content": '''
                
                [Seu conteúdo de prompt original permanece aqui...]
                
                '''},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Erro ao gerar resposta: {str(e)}"

def main():
    # Configuração da página
    st.set_page_config(page_title="🤖 Bot Luiz Lourenço", page_icon=":robot_face:")
    
    # Carrega o ícone do bot
    bot_icon = load_bot_icon()
    
    st.title("🤖 Bot Luiz Lourenço")
    st.write("Conectado à base de dados")
    
    # Inicializa cliente do Astra DB
    astra_client = AstraDBClient()
    
    # Inicializa histórico de conversa
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Exibe mensagens anteriores
    for message in st.session_state.messages:
        if message["role"] == "assistant" and bot_icon:
            # Usa a imagem como ícone para o assistente
            with st.chat_message("assistant", avatar=bot_icon):
                st.markdown(message["content"])
        else:
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
            
            # Exibe a resposta com o ícone personalizado
            if bot_icon:
                with st.chat_message("assistant", avatar=bot_icon):
                    st.markdown(response)
            else:
                with st.chat_message("assistant"):
                    st.markdown(response)

if __name__ == "__main__":
    main()
