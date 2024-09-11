# Bedrock Agent로 Chatbot 구현하기

여기에서는 Bedrock Agent를 애플리케이션에서 호출하는 방법에 대해 설명하고, code interpreter를 이용해 그래프를 그리고 longterm memory를 이용해 agent에서 이전 대화 이력을 활용하는 방법에 대해 설명합니다. 

## Bedrock Agent의 동작

- Action groups: 최대 20개의 API에 대한 orchestration을 수행할 수 있습니다.
- Knowledge bases: Agent는 Knowledge bases를 통해 필요한 문서를 조회할 수 있습니다.
- Prompt templates ([How Amazon Bedrock Agents works](https://docs.aws.amazon.com/bedrock/latest/userguide/agents-how.html))
    - Pre-processing
    - Orchestration
    - Knowledge base response generation
    - Post-processing (disabled by default)



<img width="500" alt="image" src="https://github.com/user-attachments/assets/42d19975-5ccc-4f8b-b928-f917a54241f9">


[boto3-invoke_agent](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent-runtime/client/invoke_agent.html)에 따라 agent를 호출하여 사용할 수 있습니다.



## Bedrock Agent의 생성

[Bedrock console](https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/agents)에 접속하여 [Create Agent]을 선택한 후 "tool-executor"라고 입력 후에 [Create]를 선택합니다.

<img width="600" alt="image" src="https://github.com/user-attachments/assets/b0087a23-6f96-4252-af60-432c4be32f10">

[Agent builder]에서 "Anthropic Claude 3 Sonnet"을 선택한 후에 "Instructions for the Agent"에 "너의 이름은 AWS이고 질문에 답변을 하는 AI Assistant입니다."라고 입력합니다. 

<img width="600" alt="image" src="https://github.com/user-attachments/assets/82483836-ef82-4eeb-8a7f-71b13b0b6120">


[Additional settings]에서 [Code Interpreter]를 "Enabled"로 설정합니다. 

<img width="600" alt="image" src="https://github.com/user-attachments/assets/5640bd3b-4d35-483d-a44d-e17950ac366a">

기본적으로 agent는 단일 대화 내에서만 정보를 기억합니다. 아래와 같이 메모리 기능을 활성화하면 에이전트가 최대 30일 동안 여러 세션에 걸쳐 정보를 기억할 수 있습니다. [Memory]에서 [Enable memory]를 "Enabled"로 설정합니다. 이후 [Save and exit]를 선택합니다. 

<img width="600" alt="image" src="https://github.com/user-attachments/assets/cddbdd68-60f2-46ba-9e4e-a64bc72b595f">

생성된 "tool-executor"는 Status가 "NOT_PREPARED"이므로, 아래의 오른쪽 Test에서 [Prepare]를 선택합니다. 

<img width="800" alt="image" src="https://github.com/user-attachments/assets/7313d738-78b4-4ac5-ae7d-6d92bfd07790">

[Edit in Agent Builder]를 선택한 후에 아래로 스크롤하여 [Knowledge bases]에서 [Add]를 선택합니다. 

<img width="600" alt="image" src="https://github.com/user-attachments/assets/63c69450-b480-4140-91ba-aaee0ca48fc9">

[Select knowledge base]에서 [knowledge-base.md](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/blob/main/knowledge-base.md)에서 생성한 "aws-rag"을 선택하고, "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. 모르는 질문을 받으면 솔직히 모른다고 말합니다."라고 입력합니다. 이후 [Add]를 선택합니다. 

<img width="600" alt="image" src="https://github.com/user-attachments/assets/13c26f08-536e-4757-a0e7-8fed08cb849b">

상단의 [Agent builder: tool-executor]에서 [Save]와 [Prepare]를 선택합니다.

Knowledge base에서 정상적으로 관련된 문서를 가져오는 것을 확인하기 위하여 [Test Agent]에서 "Advanced RAG에 대해 설명해주세요."라고 입력 후에 결과를 확인합니다. 

<img width="283" alt="image" src="https://github.com/user-attachments/assets/e0014841-8e2a-4a08-a0ae-38e92d35d762">

상단의 [Agent builder: tool-executor]에서 [Save and exit]를 선택합니다.

하단으로 스크롤하여 아래와 같이 [Aliases]에서 [Create]를 선택합니다. 

<img width="600" alt="image" src="https://github.com/user-attachments/assets/9f104d12-4bd8-4710-af2d-ed5017b91cf3">

아래와 같이 [Alias name]을 "latest_version"을 입력하고 [Create alias]를 선택합니다. 

<img width="600" alt="image" src="https://github.com/user-attachments/assets/f9b62833-3409-4116-852f-d84f68ff0443">


## 애플리케이션에서 생성한 Agent 호출하기

아래와 같이 'bedrock-agent-runtime'을 이용하여 client를 정의하고 invoke_agent()로 응답을 가져올 수 있습니다. 상세한 코드는 [lambda_function.py](./lambda-chat-ws/lambda_function.py)을 참조합니다. 

```python
agent_id = agent_alias_id = None
sessionId = dict() 
def run_bedrock_agent(text, connectionId, requestId, userId):
    global agent_id, agent_alias_id
    print('agent_id: ', agent_id)
    print('agent_alias_id: ', agent_alias_id)
    
    client = boto3.client(service_name='bedrock-agent')  
    if not agent_id:
        response_agent = client.list_agents(
            maxResults=10
        )
        print('response of list_agents(): ', response_agent)
        
        for summary in response_agent["agentSummaries"]:
            if summary["agentName"] == "tool-executor":
                agent_id = summary["agentId"]
                print('agent_id: ', agent_id)
                break
    
    if not agent_alias_id and agent_id:
        response_agent_alias = client.list_agent_aliases(
            agentId = agent_id,
            maxResults=10
        )
        print('response of list_agent_aliases(): ', response_agent_alias)   
        
        for summary in response_agent_alias["agentAliasSummaries"]:
            if summary["agentAliasName"] == "latest_version":
                agent_alias_id = summary["agentAliasId"]
                print('agent_alias_id: ', agent_alias_id)
                break
    
    global sessionId
    if not userId in sessionId:
        sessionId[userId] = str(uuid.uuid4())
    
    msg = msg_contents = ""
    isTyping(connectionId, requestId)  
    if agent_alias_id and agent_id:
        client_runtime = boto3.client('bedrock-agent-runtime')
        try:            
            response =  client_runtime.invoke_agent(
                agentAliasId=agent_alias_id,
                agentId=agent_id,
                inputText=text,
                sessionId=sessionId[userId],
                memoryId='memory-'+userId
            )
            print('response of invoke_agent(): ', response)
            
            response_stream = response['completion']
            
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
                    
                # files generated by code interpreter
                if 'files' in event:
                    files = event['files']['files']
                    for file in files:
                        objectName = file['name']
                        print('objectName: ', objectName)
                        contentType = file['type']
                        print('contentType: ', contentType)
                        bytes_data = file['bytes']
                                                
                        pixels = BytesIO(bytes_data)
                        pixels.seek(0, 0)
                                    
                        img_key = 'agent/contents/'+objectName
                        
                        s3_client = boto3.client('s3')  
                        response = s3_client.put_object(
                            Bucket=s3_bucket,
                            Key=img_key,
                            ContentType=contentType,
                            Body=pixels
                        )
                        print('response: ', response)
                        
                        url = path+'agent/contents/'+parse.quote(objectName)
                        print('url: ', url)
                        
                        if contentType == 'application/json':
                            msg_contents = f"\n\n<a href={url} target=_blank>{objectName}</a>"
                        elif contentType == 'application/csv':
                            msg_contents = f"\n\n<a href={url} target=_blank>{objectName}</a>"
                        else:
                            width = 600            
                            msg_contents = f'\n\n<img src=\"{url}\" alt=\"{objectName}\" width=\"{width}\">'
                            print('msg_contents: ', msg_contents)
                                                            
        except Exception as e:
            raise Exception("unexpected event.",e)
        
    return msg+msg_contents
```

## 실행 결과

메뉴에서 "Bedrock Agent"를 선택하고 "Advanced RAG에 대해 설명해주세요."와 같이 입력후 결과를 확인합니다. 

<img width="768" alt="image" src="https://github.com/user-attachments/assets/5f5e8ccb-7533-4098-98b9-16f2dcb1132e">

## Code Interpreter

Bedrock agent는 python code로된 tool을 실행하여 질문에 대한 분석을 수행하고 결과를 이미지로 제공할 수 있습니다. Code interpreter는 agent에게 python을 실행하는 sandbox 환경을 연결을 제공합니다. [AI Running Its Own Code: Agentic Code Interpreter](https://www.youtube.com/watch?v=zC_qLlm2se0)에서는 code interpreter에 대해 설명하고 있고 관련된 코드는 [Setting up and Testing an Agent for Amazon Bedrock with Code Interpreter](https://github.com/build-on-aws/agents-for-amazon-bedrock-sample-feature-notebooks/blob/main/notebooks/preview-agent-code-interpreter.ipynb)을 참고합니다.


"반복적으로 동작하는 cosine 그래프를 그려주세요."로 입력했을 때의 결과입니다.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/c1201c0c-0457-43b8-9b28-83225609151e">

"가장 최근 그래프의 값을 JSON 포맷으로 저장하세요."을 입력하면 아래와 같이 그래프의 데이터를 얻을 수 있습니다.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/9be2eb0b-134a-4d8c-b381-933374c5cc80">

json 파일을 선택하여 내용을 확인합니다.

<img width="832" alt="image" src="https://github.com/user-attachments/assets/e3f5bb42-8594-46a5-9113-71ce71158b54">




[stock_prices.csv](./contents/stock_prices.csv)을 Code Interpreter로 읽어서 처리할 경우에 아래와 같은 결과를 얻을 수 있습니다.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/8fa8ed87-4279-4583-96e4-53c7196f14b0">

이후 아래와 같이 "가장 변동량이 큰 주식의 마지막 1년의 데이터를 그래프로 그려줘."라고 입력후에 결과를 확인합니다. 

<img width="700" alt="image" src="https://github.com/user-attachments/assets/dcc2db8e-314b-45e9-b5a5-38efa23f3ffa">


## Memory

Bedrock agent는 chat history를 가지고 문맥(context)에 맞는 답변을 합니다. 여기에서는 amazon bedrock의 long term memory에 대해 설명합니다. [Solving LLM Amnesia: Cross Session Memory](https://www.youtube.com/watch?v=ZY5WXDDp9g8)에서는 사용법에 대해 설명하고 있고, [Setting up and Testing an Agent for Amazon Bedrock with Long Term Memory
](https://github.com/build-on-aws/agents-for-amazon-bedrock-sample-feature-notebooks/blob/main/notebooks/preview-agent-long-memory.ipynb)는 관련 코드를 공유하고 있습니다. 

Agent는 대화 history를 아래와 같이 저장되어 활용됩니다. 최대 저장 기간은 30일입니다. 

<img width="675" alt="image" src="https://github.com/user-attachments/assets/99a4a416-a918-4b9f-bd66-ced5b82f90f8">



## Reference

[Agents Tools & Function Calling with Amazon Bedrock (How-to)](https://www.youtube.com/watch?v=2L_XE6g3atI)

[Setting up and Testing an Agent for Amazon Bedrock with Code Interpreter](https://github.com/build-on-aws/agents-for-amazon-bedrock-sample-feature-notebooks/blob/main/notebooks/preview-agent-code-interpreter.ipynb)
