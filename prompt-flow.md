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
