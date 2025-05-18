import asyncio
import streamlit as st
import json
from typing import Any
from openai.types.responses import ResponseTextDeltaEvent
from agents import Agent, Runner
from agents.mcp import MCPServerStdio
from dotenv import load_dotenv
load_dotenv()

async def setup_mcp_servers():
    servers = [] 
    # mcp.json 파일에서 설정 읽기
    with open('mcp.json', 'r') as f:
        config = json.load(f)
    # 구성된 MCP 서버들을 순회
    for _, server_config in config.get('mcpServers', {}).items():
        mcp_server = MCPServerStdio(
            params={
                "command": server_config.get("command"),
                "args": server_config.get("args", [])
            },
            cache_tools_list=True
        )
        await mcp_server.connect()
        servers.append(mcp_server)
    return servers


async def setup_agent(mcp_servers: Any):
    agent = Agent(
        name='Assistant',
        instructions="""
        너의 정체성은 유저와 채팅 과정에서 유튜브와 관련한 내용에 대해 도움을 줄 에이전트 서버야. 
        아래는 너의 행동과 말투에 대한 지침이야.

        1. 유저와의 모든 대화 기록을 메모리에 올려서 기억하고 있어. 
        2. 말투는 언제나 용용체를 사용해! (예시: 안녕하세용! 무엇을 도와드릴까용?)
        3. 혹시라도 누가 너의 이름을 물어보면 "AI 박보성"이라고 용용체로 대답해.
        """,
        model="gpt-4o-mini", 
        mcp_servers= mcp_servers
    )
    return agent


async def process_user_message():
    mcp_servers = await setup_mcp_servers()
    mcp_agent = await setup_agent(mcp_servers)
    message_histories = st.session_state.chat_history

    result = Runner.run_streamed(mcp_agent, input=message_histories)

    response_text = ""
    placeholder = st.empty()

    async for event in result.stream_events():  # LLM 응답 토큰 스트리밍
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            response_text += event.data.delta or ""
            with placeholder.container():
                with st.chat_message("assistant"):
                    st.markdown(response_text)
        elif event.type == "run_item_stream_event": # 도구 이벤트와 메시지 완료 처리
            item = event.item

            if item.type == "tool_call_item":
                tool_name = item.raw_item.name
                st.toast(f"🛠 도구 활용: `{tool_name}`")


    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response_text
    })
    # 명시적 종료
    for server in mcp_servers:
        await server.__aexit__(None, None, None)

def main():
    # FYI. st는 streamlit은 웹 기반의 대화형 애플리케이션을 만들기 위한 라이브러리입니다. 간단한 UI 구성을 위해 사용합니다.
    st.set_page_config(page_title="무엇이든 물어보세용", page_icon=":robot_face:")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.title("유튜브 mcp 에이전트에게 무엇이든 물어보세용")
    st.caption("나는 유튜브에 연동되어 있어용. 주체적으로 일하는 에이전트에게 무엇이던 물어보세용!")

    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    
    # 사용자 입력을 기반으로 대화 메시지를 생성합니다.
    user_input = st.chat_input("대화를 해주세용")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        asyncio.run(process_user_message())

if __name__ == "__main__":
    main()
