# Bedrock Agent로 Chatbot 구현하기

여기에서는 Bedrock Agent를 애플리케이션에서 호출하는 방법에 대해 설명합니다.

## Bedrock Agent의 동작

- Action groups: 최대 20개의 API에 대한 orchestration을 수행할 수 있습니다.
- Knowledge bases: Agent는 Knowledge bases를 통해 필요한 문서를 조회할 수 있습니다.
- Prompt templates ([How Amazon Bedrock Agents works](https://docs.aws.amazon.com/bedrock/latest/userguide/agents-how.html))
    - Pre-processing
    - Orchestration
    - Knowledge base response generation
    - Post-processing (disabled by default)



<img width="550" alt="image" src="https://github.com/user-attachments/assets/42d19975-5ccc-4f8b-b928-f917a54241f9">


[boto3-invoke_agent](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent-runtime/client/invoke_agent.html)에 따라 agent를 호출하여 사용할 수 있습니다.



## Bedrock Agent의 생성

[Bedrock console](https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/agents)에 접속하여 [Create Agent]을 선택한 후 "tool-executor"라고 입력 후에 [Create]를 선택합니다.

<img width="500" alt="image" src="https://github.com/user-attachments/assets/b0087a23-6f96-4252-af60-432c4be32f10">

[Agent builder]에서 "Antropic Claude 3 Sonnet"을 선택한 후에 "Instructions for the Agent"에 "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. 모르는 질문을 받으면 솔직히 모른다고 말합니다."라고 입력합니다. 

<img width="648" alt="image" src="https://github.com/user-attachments/assets/97cc36d8-747b-43a2-92b7-ca98cd180c9b">

[Additional settings]에서 [Code Interpreter]를 "Enabled"로 설정합니다. 

<img width="616" alt="image" src="https://github.com/user-attachments/assets/5640bd3b-4d35-483d-a44d-e17950ac366a">

기본적으로 agent는 단일 대화 내에서만 정보를 기억합니다. 아래와 같이 메모리 기능을 활성화하면 에이전트가 최대 30일 동안 여러 세션에 걸쳐 정보를 기억할 수 있습니다. [Memory]에서 [Enable memory]를 "Enabled"로 설정합니다. 

![noname](https://github.com/user-attachments/assets/cddbdd68-60f2-46ba-9e4e-a64bc72b595f)

## 구현 코드

아래와 같이 'bedrock-agent-runtime'을 이용하여 client를 정의하고 invoke_agent()로 응답을 가져올 수 있습니다. 


```python
agent_alias_id = None
agent_id = None
def run_bedrock_agent(text, connectionId, requestId, userId):
    global agent_id, agent_alias_id
    
    client = boto3.client(service_name='bedrock-agent')  
    if not agent_id:
        response_agent = client.list_agents(
            maxResults=10
        )
        
        for summary in response_agent["agentSummaries"]:
            if summary["agentName"] == "tool-executor":
                agent_id = summary["agentId"]
                break
    
    if not agent_alias_id and agent_id:
        response_agent_alias = client.list_agent_aliases(
            agentId = agent_id,
            maxResults=10
        )
        
        for summary in response_agent_alias["agentAliasSummaries"]:
            if summary["agentAliasName"] == "latest_version":
                agent_alias_id = summary["agentAliasId"]
                break
    
    msg = ""    
    if agent_alias_id and agent_id:
        client_runtime = boto3.client('bedrock-agent-runtime')
        try:
            response =  client_runtime.invoke_agent(
                agentAliasId=agent_alias_id,
                agentId=agent_id,
                inputText=text,
                sessionId='session-'+userId,
                # memoryId='memory-01'
            )
            
            response_stream = response['completion']
            
            for event in response_stream:
                chunk = event.get('chunk')
                if chunk:
                    msg += chunk.get('bytes').decode()
                        
                    result = {
                        'request_id': requestId,
                        'msg': msg,
                        'status': 'proceeding'
                    }
                    sendMessage(connectionId, result)
                                    
        except Exception as e:
            raise Exception("unexpected event.",e)
        
    return msg
```

## 실행 결과

메뉴에서 "Bedrock Agent"를 선택하고 아래와 같이 테스트해 볼 수 있습니다.

<img width="785" alt="image" src="https://github.com/user-attachments/assets/4d58d2f7-7d95-4477-806f-309be24735e9">



## Code Interpreter

Bedrock agent는 python code로된 tool을 실행하여 질문에 대한 분석을 수행하고 결과를 이미지로 제공할 수 있습니다. Code interpreter는 agent에게 python을 실행하는 sandbox 환경을 연결을 제공합니다. 

[AI Running Its Own Code: Agentic Code Interpreter](https://www.youtube.com/watch?v=zC_qLlm2se0)에서는 code interpreter에 대해 설명하고 있고 관련된 코드는 [Setting up and Testing an Agent for Amazon Bedrock with Code Interpreter](https://github.com/build-on-aws/agents-for-amazon-bedrock-sample-feature-notebooks/blob/main/notebooks/preview-agent-code-interpreter.ipynb)을 참고합니다.


"반복적으로 동작하는 cosine 그래프를 그려주세요."로 입력했을 때의 결과입니다.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/c1201c0c-0457-43b8-9b28-83225609151e">

"가장 최근 그래프의 값을 JSON 포맷으로 저장하세요."을 입력하면 아래와 같이 그래프의 데이터를 얻을 수 있습니다.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/9be2eb0b-134a-4d8c-b381-933374c5cc80">

json 파일을 선택하여 내용을 확인합니다.

<img width="832" alt="image" src="https://github.com/user-attachments/assets/e3f5bb42-8594-46a5-9113-71ce71158b54">




## Memory

Bedrock agent는 chat history를 가지고 문맥(context)에 맞는 답변을 합니다. 여기에서는 amazon bedrock의 long term memory에 대해 설명합니다. [Solving LLM Amnesia: Cross Session Memory](https://www.youtube.com/watch?v=ZY5WXDDp9g8)에서는 사용법에 대해 설명하고 있고, [Setting up and Testing an Agent for Amazon Bedrock with Long Term Memory
](https://github.com/build-on-aws/agents-for-amazon-bedrock-sample-feature-notebooks/blob/main/notebooks/preview-agent-long-memory.ipynb)는 관련 코드를 공유하고 있습니다. 

Agent는 대화 history를 아래와 같이 저장되어 활용됩니다. 최대 저장 기간은 30일입니다. 

<img width="675" alt="image" src="https://github.com/user-attachments/assets/99a4a416-a918-4b9f-bd66-ced5b82f90f8">


