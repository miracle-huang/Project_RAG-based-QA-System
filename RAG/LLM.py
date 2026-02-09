from typing import Dict, List, Optional, Tuple, Union

'''
大语言模型模块 - 提供对话生成能力
'''

PROMPT_TEMPLATE = dict(
    # 中文 Prompt 模板
    RAG_PROMPT_TEMPALTE="""使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:""",
    InternLM_PROMPT_TEMPALTE="""先对上下文进行内容总结,再使用上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:""",
    # 日本語 Prompt テンプレート
    RAG_PROMPT_TEMPALTE_JA="""以下のコンテキストを使用してユーザーの質問に回答してください。答えがわからない場合は、わからないと言ってください。常に日本語で回答してください。
        質問: {question}
        参考コンテキスト：
        ···
        {context}
        ···
        与えられたコンテキストで回答できない場合は、データベースにその内容がないため、わからないと回答してください。
        回答:""",
    InternLM_PROMPT_TEMPALTE_JA="""まずコンテキストの内容を要約し、次にコンテキストを使用してユーザーの質問に回答してください。答えがわからない場合は、わからないと言ってください。常に日本語で回答してください。
        質問: {question}
        参考コンテキスト：
        ···
        {context}
        ···
        与えられたコンテキストで回答できない場合は、データベースにその内容がないため、わからないと回答してください。
        回答:"""
)

class BaseModel:
    def __init__(self, path: str = '') -> None:
        self.path = path

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        pass

    def load_model(self):
        pass

class OpenAIChat(BaseModel):
    def __init__(self, path: str = '', model: str = "gpt-3.5-turbo-1106") -> None:
        super().__init__(path)
        self.model = model

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        from openai import OpenAI
        client = OpenAI()   
        history.append({'role': 'user', 'content': PROMPT_TEMPLATE['RAG_PROMPT_TEMPALTE'].format(question=prompt, context=content)})
        response = client.chat.completions.create(
            model=self.model,
            messages=history,
            max_tokens=150,
            temperature=0.1
        )
        return response.choices[0].message.content

class InternLMChat(BaseModel):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)
        self.load_model()

    def chat(self, prompt: str, history: List = [], content: str='') -> str:
        prompt = PROMPT_TEMPLATE['InternLM_PROMPT_TEMPALTE'].format(question=prompt, context=content)
        response, history = self.model.chat(self.tokenizer, prompt, history)
        return response


    def load_model(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.path, torch_dtype=torch.float16, trust_remote_code=True).cuda()


class ZhipuChat(BaseModel):
    """
    使用智谱AI GLM模型进行对话
    """
    def __init__(self, path: str = '', model: str = "glm-4-flash") -> None:
        super().__init__(path)
        self.model = model
        import os
        from zhipuai import ZhipuAI
        self.client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))

    def chat(self, prompt: str, history: List[dict], content: str, lang: str = 'zh') -> str:
        """
        与智谱AI进行对话
        Args:
            prompt: 用户问题
            history: 对话历史
            content: RAG检索到的上下文内容
            lang: 语言选择，'zh'为中文，'ja'为日文
        """
        # 根据语言选择对应的 Prompt 模板
        if lang == 'ja':
            template_key = 'RAG_PROMPT_TEMPALTE_JA'
        else:
            template_key = 'RAG_PROMPT_TEMPALTE'
        
        history.append({
            'role': 'user', 
            'content': PROMPT_TEMPLATE[template_key].format(question=prompt, context=content)
        })
        response = self.client.chat.completions.create(
            model=self.model,
            messages=history,
            max_tokens=500,
            temperature=0.1
        )
        return response.choices[0].message.content