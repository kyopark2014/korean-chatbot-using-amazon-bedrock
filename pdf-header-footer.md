# PDF에서 Header와 Footer 처리

여기에서는 PDF를 읽을때에 header와 footer를 제외한 본문만을 가져오는 방법에 대해 설명합니다.



```python
doc = fitz.open(fname)
page = doc[0]
rect = page.rect
height = 50
clip = fitz.Rect(0, height, rect.width, rect.height-height)
text = page.get_text(clip=clip)
```


## Reference 

[python - read pdf ignoring header and footer](https://stackoverflow.com/questions/68082761/python-read-pdf-ignoring-header-and-footer)
