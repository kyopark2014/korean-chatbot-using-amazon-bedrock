# RAG에서 Grading 활용하기 

RAG는 지식저장소에서 관련된 문서(Relevant doc)을 가져옵니다. 이때 RAG의 성능을 향상시키기 위해 관련된 문서가 실제로 관련되었는지 확인하는 절차가 필요합니다. 이러한 방법에서 Rerank나 [priority search](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/tree/main?tab=readme-ov-file#priority-search-%EA%B4%80%EB%A0%A8%EB%8F%84-%EA%B8%B0%EC%A4%80-%EB%AC%B8%EC%84%9C-%EC%84%A0%ED%83%9D)를 활용할 수 있습니다. 여기에서는 LLM으로 관련도를 확인하는 방법을 설명합니다.
