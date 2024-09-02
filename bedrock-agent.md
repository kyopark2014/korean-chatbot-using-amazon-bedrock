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



<img width="500" alt="image" src="https://github.com/user-attachments/assets/42d19975-5ccc-4f8b-b928-f917a54241f9">


[boto3-invoke_agent](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent-runtime/client/invoke_agent.html)에 따라 Agent를 호출하여 사용할 수 있습니다.

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

Bedrock agent는 python code로된 tool을 실행하여 질문에 대한 분석을 수행하고 결과를 이미지로 제공할 수 있습니다. Code interpreter는 agent에게 python을 실행하는 sandbox 환경을 연결을 제공합니다. 



"반복적으로 동작하는 cosine 그래프를 그려주세요."로 입력했을때의 결과입니다.

![noname](https://github.com/user-attachments/assets/c1201c0c-0457-43b8-9b28-83225609151e)



[AI Running Its Own Code: Agentic Code Interpreter](https://www.youtube.com/watch?v=zC_qLlm2se0)에서는 code interpreter에 대해 설명하고 있고 관련된 코드는 [Setting up and Testing an Agent for Amazon Bedrock with Code Interpreter](https://github.com/build-on-aws/agents-for-amazon-bedrock-sample-feature-notebooks/blob/main/notebooks/preview-agent-code-interpreter.ipynb)을 참고합니다.


## Memory

[Solving LLM Amnesia: Cross Session Memory](https://www.youtube.com/watch?v=ZY5WXDDp9g8)와 [Setting up and Testing an Agent for Amazon Bedrock with Long Term Memory
](https://github.com/build-on-aws/agents-for-amazon-bedrock-sample-feature-notebooks/blob/main/notebooks/preview-agent-long-memory.ipynb)을 참조합니다.

대화 history는 아래와 같이 저장되어 활용됩니다.

<img width="675" alt="image" src="https://github.com/user-attachments/assets/99a4a416-a918-4b9f-bd66-ced5b82f90f8">
