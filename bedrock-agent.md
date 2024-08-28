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
def run_bedrock_agent(text, connectionId, requestId):
    client_runtime = boto3.client('bedrock-agent-runtime')
    response =  client_runtime.invoke_agent(
        agentAliasId='CEXQFZT1EL',
        agentId='2SI1ONTVMW',
        inputText=text,
        sessionId='session-01',
        # memoryId='memory-01'
    )
    print('response of invoke_agent(): ', response)
    
    response_stream = response['completion']
    
    msg = ""
    try:
        for event in response_stream:
            chunk = event.get('chunk')
            if chunk:
                msg += chunk.get('bytes').decode()
                print('event: ', chunk.get('bytes').decode())
                
                result = {
                    'request_id': requestId,
                    'msg': msg,
                    'status': 'proceeding'
                }
                #print('result: ', json.dumps(result))
                sendMessage(connectionId, result)
                                
    except Exception as e:
        raise Exception("unexpected event.",e)
        
    return msg
```

## 실행 결과

메뉴에서 "Bedrock Agent"를 선택하고 아래와 같이 테스트해 볼 수 있습니다.

<img width="785" alt="image" src="https://github.com/user-attachments/assets/4d58d2f7-7d95-4477-806f-309be24735e9">

