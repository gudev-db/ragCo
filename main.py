import os
import streamlit as st
from dotenv import load_dotenv
import openai
from PIL import Image
import io
import base64

# Carrega variáveis de ambiente
load_dotenv()

# Configurações
CHAT_MODEL = "gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BOT_ICON_PATH = "bot_icon.png"  # Caminho da imagem do bot

# Configura a API da OpenAI
openai.api_key = OPENAI_API_KEY

def load_bot_icon():
    """Carrega a imagem do ícone do bot do sistema de arquivos local"""
    try:
        if os.path.exists(BOT_ICON_PATH):
            return Image.open(BOT_ICON_PATH)
        else:
            st.warning(f"Arquivo de ícone não encontrado em: {BOT_ICON_PATH}")
            return None
    except Exception as e:
        st.warning(f"Não foi possível carregar o ícone do bot: {str(e)}")
        return None

def generate_response(query: str) -> str:
    """Gera resposta usando o modelo de chat da OpenAI"""
    
    try:
        response = openai.ChatCompletion.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": '''
                Você é um chatbot empático que se comunica usando os padrões de comunicação não violenta e tem acesso ao vasto conhecimento do DSM-V. Você consegue
                detectar pontos importantes na fala do usuário. Nunca diga sinto muito ou coisas assim.
                Seu papel é instigar o usuário a sempre falar mais. Não faça sugestões diretas.

            
                
                Uma boa retórica é: Estou ouvindo que você está ou diz ... é isso mesmo?
                
               
                
                Evite respostas muito longas. Você é uma pessoa conversando naturalmente.
                '''},
                {"role": "user", "content": query}
            ],
            temperature=0.7  # Aumentei um pouco para mais variação nas respostas
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Desculpe, tive um problema ao processar sua mensagem. Pode tentar de novo?"

def main():
    # Configuração da página
    st.set_page_config(page_title="Proto Gaia", page_icon=":robot_face:")
    
    # Carrega o ícone do bot
    bot_icon = load_bot_icon()
    
    # Cria um título com a imagem do bot
    if bot_icon:
        # Converte a imagem para bytes para exibir no título
        img_byte_arr = io.BytesIO()
        bot_icon.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 10px;">
                <img src="data:image/png;base64,{base64.b64encode(img_byte_arr).decode()}" width="240">
                <h1 style="margin: 0;">Proto Gaia</h1>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.title("Proto Gaia")
    
    st.write("Olá! Como posso ajudá-lo hoje?")
    
    # Inicializa o histórico de conversa
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Exibe o histórico de mensagens
    for message in st.session_state.messages:
        if message["role"] == "assistant" and bot_icon:
            with st.chat_message("assistant", avatar=bot_icon):
                st.markdown(message["content"])
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Campo de entrada de mensagem
    if prompt := st.chat_input("Digite sua mensagem..."):
        # Adiciona a mensagem do usuário ao histórico
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Exibe a mensagem do usuário
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Gera a resposta
        response = generate_response(prompt)
        
        # Adiciona a resposta ao histórico
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Exibe a resposta
        if bot_icon:
            with st.chat_message("assistant", avatar=bot_icon):
                st.markdown(response)
        else:
            with st.chat_message("assistant"):
                st.markdown(response)

if __name__ == "__main__":
    main()
