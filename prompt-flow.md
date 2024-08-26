# Prompt Flow

여기에서는 Prompt Flow를 이용하여 RAG가 적용된 chatbot을 만드는것을 설명합니다.

[프롬프트 플로우 실행 코드 샘플](https://docs.aws.amazon.com/ko_kr/bedrock/latest/userguide/flows-code-ex.html)를 참조하여 구현합니다.


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

## Prompt Flow의 실행

[boto3-invoke_flow](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent-runtime/client/invoke_flow.html)와 [AWS Doc: InvokeFlow](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_InvokeFlow.html)에서는 아래와 같이 입력문을 표현하고 있습니다.

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

아래의 경우에서는 nodeName은 "FlowInputNode"이고, nodeOutputName은 "document"입니다. 

![image](https://github.com/user-attachments/assets/112a8d72-956b-485d-a50a-a252e01410a3)

