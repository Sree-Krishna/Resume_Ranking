import re
from llama_index.core.schema import TransformComponent

class TextCleaner(TransformComponent):
    def __call__(self, nodes, **kwargs):
        for node in nodes:
            node.text = re.sub(r"[^0-9A-Za-z ]", "", node.text)
            node.text = node.text.lower()
        return nodes
    
def clean_job(text):
    text = re.sub(r"[^0-9A-Za-z ]", "", text)
    return text.lower()