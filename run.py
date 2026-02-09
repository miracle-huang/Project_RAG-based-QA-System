from RAG.VectorBase import VectorStore
from RAG.utils import ReadFiles
from RAG.LLM import OpenAIChat, InternLMChat, ZhipuChat
from RAG.Embeddings import JinaEmbedding, ZhipuEmbedding


# ============ 本地模型使用示例 ============
# 使用 Jina Embedding（本地） + InternLM（本地）
# 取消以下代码的注释即可使用

# # 加载本地向量数据库
# vector = VectorStore()
# vector.load_vector('./storage')

# # 使用 Jina 本地 Embedding 模型（首次运行会自动下载）
# embedding = JinaEmbedding("jinaai/jina-embeddings-v2-base-zh")

# # 使用 InternLM 本地 LLM 模型（需要先下载到本地）
# # 模型路径请修改为你实际下载的路径
# INTERNLM_MODEL_PATH = "E:/models/internlm2-chat-1_8b"  # 修改为你的模型路径
# chat = InternLMChat(path=INTERNLM_MODEL_PATH)

# # 提问
# question = '逆向纠错的原理是什么？'
# content = vector.query(question, EmbeddingModel=embedding, k=1)[0]
# print(chat.chat(question, [], content))

# ============ API 模型使用示例 ============
# 没有保存数据库
# docs = ReadFiles('./data').get_content(max_token_len=600, cover_content=150) # 获得data目录下的所有文件内容并分割
# vector = VectorStore(docs)
# embedding = ZhipuEmbedding() # 创建EmbeddingModel
# vector.get_vector(EmbeddingModel=embedding)
# vector.persist(path='storage') # 将向量和文档内容保存到storage目录下，下次再用就可以直接加载本地的数据库

# vector.load_vector('./storage') # 加载本地的数据库

# question = '正向扫描的原理是什么？'

# content = vector.query(question, model='zhipu', k=1)[0]
# chat = OpenAIChat(model='gpt-3.5-turbo-1106')
# print(chat.chat(question, [], content))


# ============ 首次使用：重新生成向量数据库 ============

print("正在读取文档...")
docs = ReadFiles('./data').get_content(max_token_len=600, cover_content=150)

print("正在使用智谱AI生成向量（这可能需要几分钟）...")
embedding = ZhipuEmbedding()
vector = VectorStore(docs)
vector.get_vector(EmbeddingModel=embedding)

print("正在保存向量数据库...")
vector.persist(path='storage')
print("向量数据库已保存到 storage 目录！")

# ============ 查询测试 ============
question = '下人最初在罗生门下“走投无路”的具体原因是什么？'
print(f"\n问题: {question}")

content = vector.query(question, EmbeddingModel=embedding, k=1)[0]
print(f"检索到的相关内容: {content[:200]}...")

# ============ 使用智谱AI GLM模型 ============
chat = ZhipuChat(model='glm-4-flash')
print(f"\n智谱AI GLM回答:")
print(chat.chat(question, [], content))

# ============ 追加测试 ============
print("\n" + "="*50)
question2 = '老婆关于“拔死人头发”的解释，为什么会成为下人后来行为转变的关键触发点？'
print(f"\n问题: {question2}")

content2 = vector.query(question2, EmbeddingModel=embedding, k=1)[0]
print(f"检索到的相关内容: {content2[:200]}...")
print(f"\n智谱AI GLM 回答:")
print(chat.chat(question2, [], content2))

# ============ 使用OpenAI模型（已注释） ============
# chat = OpenAIChat(model='gpt-3.5-turbo-1106')
# print(f"\nGPT回答:")
# print(chat.chat(question, [], content))

