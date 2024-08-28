# Prompt Flow 활용하기

Prompt Flow를 이용하면 prompt flow builder를 이용하여 손쉽게 chatbot을 만들 수 있습니다. 여기에서는 Anthropic의 Claude Sonnet를 이용하여 아래와 같은 chatbot을 구현하고, 애플리케이션에서 활용하는 방법을 설명합니다. 

1) 이름을 가진 chatbot 구현하기: "AWS"라는 이름을 가지는 chatbot을 구현하여 prompt flow의 동작을 이해합니다.
2) RAG를 활용하는 Chatbot 구현하기: No code로 RAG를 구현하기 위하여 [Knowledge Base](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/blob/main/knowledge-base.md)를 활용합니다. 

## Prompt Flow로 이름을 가지는 Chatbot 구현하기

Prompt flow를 이용하면 별도 코딩없이 Prompt, RAG, Lambda등을 chatbot에 통합할 수 있습니다. 여기에서는 prompt flow의 동작 방식을 설명하기 위하여, "AWS"라는 이름을 가지는 간단한 chatbot을 구현하는 것을 설명합니다.

1) [Prompt flow console](https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/prompt-flows)에 접속하여 [Create prompt flow] 선택하고 아래와 같이 입력 후에 [Create]를 선택합니다.

![noname](https://github.com/user-attachments/assets/9af83873-06d2-4ba8-b52d-e70c4fa72eac)

2) 왼쪽의 [Nodes]에서 [Prompts[를 선택한 후, 아래와 같이 "Flow input", "Flow output" 노드와 연결해 줍니다.

![image](https://github.com/user-attachments/assets/b279580e-3272-4633-93f6-32f74adf8a19)

3) Prompts 노드는 아래와 같이 입력합니다. 여기서는 [Node name]로 "aws_chatbot", [Define in node]을 선택하고, Model은 "Claude 3 Sonnet"을 지정하였으며, Prompt에는 "너의 이름은 AWS이고 질문에 답변을 하는 AI Assistant입니다. 다음의 {{input}}에 대해 구체적인 세부 정보를 충분히 제공합니다."라고 입력하였습니다. [Save] 버튼을 선택하여 작업한 내용을 저장합니다. 

![noname](https://github.com/user-attachments/assets/eba3287e-d174-4d2e-8503-c04b5c87aec7)


4) 오른쪽의 [Test Prompt flow]의 입력창에 "안녕"이라고 입력하면, 아래와 같이 "AWS"라는 이름을 가지는 chatbot이 생성되었음을 알 수 있습니다. 

![image](https://github.com/user-attachments/assets/0544b16a-f142-425c-97db-0f8bc971c17a)

5) 다시 [Prompt flow console](https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/prompt-flows)에서 생성한 "aws_bot"을 선택한 후에 아래와 같이 [Publish version]을 선택하면 "Version 1"이 생성됩니다.
   
![noname](https://github.com/user-attachments/assets/26c5824e-a5d8-4693-b9d6-6243e03c570b)

6) 아래로 스크롤하여 [Create alias]를 선택합니다.

![noname](https://github.com/user-attachments/assets/f1300dda-ae80-480a-b835-9dfc6c168e71)

이후 아래와 같이 "Alias name"으로 "aws"라고 입력한 후에 [Create Alias]를 선택합니다. 

![noname](https://github.com/user-attachments/assets/eb786b75-83bc-4e80-a5ee-e198732c7abe)

6) 아래와 같이 [Prompt flow ARN]을 복사합니다.

<img width="500" alt="image" src="https://github.com/user-attachments/assets/8fb43e61-5259-444c-bd1c-d17b82580e37">




## 애플리케이션에서 Prompt Flow 활용하기

[프롬프트 플로우 실행 코드 샘플](https://docs.aws.amazon.com/ko_kr/bedrock/latest/userguide/flows-code-ex.html)를 참조하여 구현합니다. 상세한 코드는 [lambda_function.py](./lambda-chat-ws/lambda_function.py)을 참조합니다.

### 각종 Parameter

- [URI Request Parameters](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_InvokeFlow.html#API_agent-runtime_InvokeFlow_RequestSyntax)와 같이  flowAliasIdentifier와 flowIdentifier는 arn입니다. 

- [boto3-invoke_flow](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent-runtime/client/invoke_flow.html)와 [AWS Doc: InvokeFlow](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_InvokeFlow.html)와 같이 prompt flow를 실행하기 위하여 아래와 같은 입력문이 필요합니다.
  
```python
POST /flows/flowIdentifier/aliases/flowAliasIdentifier HTTP/1.1
Content-type: application/json

{
   "inputs": [ 
      { 
         "content": { ... },
         "nodeName": "string",
         "nodeOutputName": "string"
      }
   ]
}
```

여기서, JSON에 필요한 값들은 아래와 같습니다. 

- nodeName: flow input node의 이름

- nodeOutputName: prompt flow의 시작인 input node의 output의 이름

현재 작성한 prompt flow는 nodeName이 "FlowInputNode"이고, nodeOutputName은 "document"입니다. 


### 애플리케이션에서 Prompt Flow를 이용하여 Chatbot 활용하기

아래와 같이 flow_id로 "Prompt flow ARN"을 붙여넣기하고, flow_alias는 생성할 때에 사용한 "aws"을 입력합니다. "flowAliasIdentifier"는 아래와 같이 list_flow_aliases()에서 alias를 검색하여 확인합니다. invoke_flow()을 이용하여 prompt flow에 입력문을 전달후 결과를 얻습니다. 

```python
flow_id = [Prompt flow ARN]
flow_alias = "aws"

def run_prompt_flow(text, connectionId, requestId):    
    client = boto3.client(service_name='bedrock-agent')   
    
    # get flow alias arn
    response_flow_aliases = client.list_flow_aliases(
        flowIdentifier=flow_id
    )
    print('response_flow_aliases: ', response_flow_aliases)
    flowAliasIdentifier = ""
    flowAlias = response_flow_aliases["flowAliasSummaries"]
    for alias in flowAlias:
        print('alias: ', alias)
        if alias['name'] == flow_alias:
            flowAliasIdentifier = alias['arn']
            print('flowAliasIdentifier: ', flowAliasIdentifier)
            break
    
    client_runtime = boto3.client('bedrock-agent-runtime')
    response = client_runtime.invoke_flow(
        flowIdentifier=flow_id,
        flowAliasIdentifier=flowAliasIdentifier,
        inputs=[
            {
                "content": {
                    "document": text,
                },
                "nodeName": "FlowInputNode",
                "nodeOutputName": "document"
            }
        ]
    )
    
    response_stream = response['responseStream']
    try:
        result = {}
        for event in response_stream:
            print('event: ', event)
            result.update(event)

        if result['flowCompletionEvent']['completionReason'] == 'SUCCESS':
            msg = readStreamMsg(connectionId, requestId, result['flowOutputEvent']['content']['document'])
        else:
            print("The prompt flow invocation completed because of the following reason:", result['flowCompletionEvent']['completionReason'])
    except Exception as e:
        raise Exception("unexpected event.",e)

    return msg
```

### 실행결과

애플리케이션을 실행하고 아래와 같이 "안녕"이라고 입력하고 결과를 확인합니다. Prompt flow를 이용하여 "AWS"라는 이름의 chatbot을 생성할 수 있었습니다.

![image](https://github.com/user-attachments/assets/38f38b74-5bcc-46e9-8019-7e8581e40465)


## RAG를 활용하는 Chatbot 구현하기

### Knowdge Base로 RAG 구현

[knowledge-base.md](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/blob/main/knowledge-base.md)와 같이 Amazon Bedrock에서는 완전관리형 RAG를 쉽게 구현할 수 있습니다. 

### Prompt Flow를 이용해 No Code로 RAG 활용하기

1) [Prompt Flow Console](https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/prompt-flows)에서 [Create prompt flow]를 선택한 후에 아래와 같이 이름을 지정합니다.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/e9715f2f-7596-4a85-98f3-8ae084f00cee">


2) 생성된 Prompt Flow에 선택한 후에 [Edit in prompt flow builder]을 선택합니다.

3) Node에서 "Knowledge Base"를 드레그인 하여 아래와 같이 배치합니다.

<img width="730" alt="image" src="https://github.com/user-attachments/assets/74f4ed1b-131a-462e-95f0-a9f366f81210">

4) Knowledge Base를 아래와 같이 설정합니다.

<img width="300" alt="image" src="https://github.com/user-attachments/assets/569fea43-3c28-427a-a3a6-0701def85b50">

5) RAG 동작을 아래와 같이 확인할 수 있습니다.

<img width="334" alt="image" src="https://github.com/user-attachments/assets/60be179f-a5f3-4574-a49e-fb851597aad3">

### 애플리케이션에서 RAG 활용하기 

[lambda_function.py](./lambda-chat-ws/lambda_function.py)의 run_RAG_prompt_flow()와 같이 rag_flow_id와 rag_flow_alias를 이용해 생성한 prompt flow를 활용할 수 있습니다.

이때의 실행 결과는 아래와 같습니다.

<img width="785" alt="image" src="https://github.com/user-attachments/assets/9e4421ab-0862-47bc-9ad9-90325d7e2d91">







<!--

## Role

```java
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "bedrock:GetFlow",
            "Resource": [
                "arn:aws:bedrock:us-west-2:677146750822:flow/TQE3MT9IQO"
            ]
        },
        {
            "Effect": "Allow",
            "Action": "bedrock:GetPrompt",
            "Resource": [
                "arn:aws:bedrock:us-west-2:677146750822:prompt/VDZVA1UNJG"
            ]
        },
        {
            "Effect": "Allow",
            "Action": "bedrock:InvokeModel",
            "Resource": [
                "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0"
            ]
        }
    ]
}
```
--> 

