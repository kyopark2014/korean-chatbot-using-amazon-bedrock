# Hybrid Search

- Semantic search(의미 검색)
  - 사용자의 질문이 정확한 키워드와 직접적으로 관련되지 않더라도 사람처럼 사용자의 질문을 이해할 수 있기 때문에 널리 사용됩니다.
    - 사용자의 질문은 때로는 질문의 keyword와 직접적으로 연관되지 않을수 있습니다. 예) "이 영화 재미있어요?"라는 질문은 영화의 리뷰, 평점에 대한 콘텐츠를 제공해달라는 의미입니다.
    - 콘텐츠에 있는 답변과 직접적으로 관련되지 않은 사용자의 질문도 이해할 수 있습니다.
  - 텍스트의 의미를 표현하는 데 사용되는 워드 임베딩의 품질에 의존적입니다.
  - 관련자료: [Knowledge Bases for Amazon Bedrock now supports hybrid search](https://aws.amazon.com/ko/blogs/machine-learning/knowledge-bases-for-amazon-bedrock-now-supports-hybrid-search/)
      
- 키워드 검색과 결합하면 다음과 같은 이점이 있습니다
  - 키워드 검색으로 직접적인 매칭 결과를 얻을 수 있습니다.
  - 의미 검색으로 키워드가 없더라도 관련된 의미 있는 결과를 얻을 수 있습니다.
  - 두 가지 검색 결과를 통합하여 더 포괄적이고 정확한 결과를 제공할 수 있습니다.

