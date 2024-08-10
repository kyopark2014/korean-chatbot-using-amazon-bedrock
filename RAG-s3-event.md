# S3 Event를 이용한 RAG 등록 및 이미지 추출

## S3 Event

등록이 필요한 Event Type에서 CREATED_PUT, REMOVED_DELETE, CREATED_COMPLETE_MULTIPART_UPLOAD가 있습니다. 여기서 CREATED_COMPLETE_MULTIPART_UPLOAD은 대용량 파일 업로드시 발생하는 event 입니다.

```python
const s3PutEventSource = new lambdaEventSources.S3EventSource(s3Bucket, {
    events: [
      s3.EventType.OBJECT_CREATED_PUT,
      s3.EventType.OBJECT_REMOVED_DELETE,
      s3.EventType.OBJECT_CREATED_COMPLETE_MULTIPART_UPLOAD
    ],
    filters: [
      { prefix: s3_prefix+'/' },
    ]
  });
  lambdaS3eventManager.addEventSource(s3PutEventSource);
```


