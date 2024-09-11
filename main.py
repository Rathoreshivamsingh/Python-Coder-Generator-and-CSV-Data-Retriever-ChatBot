from typing import Any
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent,AgentExecutor
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_csv_agent
from langchain_core.tools import Tool
from langchain_experimental.tools import PythonAstREPLTool
from langchain_experimental.tools import PythonREPLTool
import logging

load_dotenv()

print("Start...")

instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
        """
base_prompt = hub.pull("langchain-ai/react-agent-template")
prompt = base_prompt.partial(instructions=instructions)

tool1 = [PythonAstREPLTool()]
python_agent = create_react_agent(
        prompt=prompt,
        llm=ChatGroq(temperature=0, model="mixtral-8x7b-32768"),
        tools=tool1,
    )

python_agent_executor = AgentExecutor(agent=python_agent, tools=tool1, verbose=True)
# final  = python_agent_executor.invoke( {"input": "Write a python program that do swap of two numbers"})
# print(final)

tool2 = [PythonREPLTool()]
csv_agent = create_csv_agent(
        llm=ChatGroq(temperature=0, model="mixtral-8x7b-32768"),
        path="episode_info.csv",
        allow_dangerous_code=True,
        verbose=True,
        tools = tool2
    )


# csv_agent_executor= AgentExecutor(agent=csv_agent,tools=tool2,allow_dangerous_code=True,verbose=True)
# finall= csv_agent.invoke( {"input": "create a list of season which have 24 episodes"})
# print(finall)

####################################### Router Grand Agent@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

def python_agent_executor_wrapper(original_prompt: str) -> dict[str, Any]:
        return python_agent_executor.invoke({"input": original_prompt})

tools = [
        Tool(
            name="Python Agent",
            func=python_agent_executor_wrapper,
            description="""useful when you need to transform natural language to python and execute the python code,
                          returning the results of the code execution
                          DOES NOT ACCEPT CODE AS INPUT""",
        ),
        Tool(
            name="CSV Agent",
            func=csv_agent.invoke,
            description="""useful when you need to answer question over episode_info.csv file,
                         takes an input the entire question and returns the answer after running pandas calculations""",
        ),
    ]

prompt = base_prompt.partial(instructions="")
grand_agent = create_react_agent(
        prompt=prompt,
        llm=ChatGroq(temperature=0, model="mixtral-8x7b-32768"),
        tools=tools,
    )
grand_agent_executor = AgentExecutor(agent=grand_agent, tools=tools, verbose=True)
logging.info("Invoking grand agent for CSV query...")

print(
        grand_agent_executor.invoke(
            {
                "input": "Which season have the most episodes",
            }
        )
    )
logging.info("CSV query executed successfully.")
print(
        grand_agent_executor.invoke(
            {
                "input": "Write a code that do basic addition ",
            }
        )
    )


