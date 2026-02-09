#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
日本語ドキュメント用 RAG 例

このスクリプトは日本語のドキュメントを読み込み、
智谱AI を使用してベクトル化と質問応答を行います。
"""

from RAG.VectorBase import VectorStore
from RAG.utils import ReadFiles
from RAG.LLM import ZhipuChat
from RAG.Embeddings import ZhipuEmbedding


def main():
    # ============ 日本語ドキュメントを読み込む ============
    print("ドキュメントを読み込んでいます...")
    docs = ReadFiles('./data_ja').get_content(max_token_len=600, cover_content=150)
    print(f"読み込んだドキュメント数: {len(docs)}")

    # ============ 智谱AI でベクトルを生成 ============
    print("智谱AI でベクトルを生成しています（数分かかる場合があります）...")
    embedding = ZhipuEmbedding()
    vector = VectorStore(docs)
    vector.get_vector(EmbeddingModel=embedding)

    # ============ ベクトルデータベースを保存 ============
    print("ベクトルデータベースを保存しています...")
    vector.persist(path='storage_ja')
    print("ベクトルデータベースが storage_ja ディレクトリに保存されました！")

    # ============ クエリテスト ============
    question = '下人は、羅生門の下で、なぜ「行き場のない状態」に追い込まれていたのか。本文の内容に基づいて説明せよ。'
    print(f"\n質問: {question}")

    content = vector.query(question, EmbeddingModel=embedding, k=1)[0]
    print(f"検索された関連コンテンツ: {content[:200]}...")

    # ============ 智谱AI GLM モデルで回答 ============
    chat = ZhipuChat(model='glm-4-flash')
    print(f"\n智谱AI GLM 回答:")
    # lang='ja' を指定して日本語プロンプトを使用
    print(chat.chat(question, [], content, lang='ja'))

    # ============ 追加のテスト質問 ============
    print("\n" + "="*50)
    question2 = '老婆が「死人の髪の毛を抜く」行為を正当化した理由は何か。また、その説明が、下人のその後の行動にどのような影響を与えたのかを、本文に即して述べよ。'
    print(f"\n質問: {question2}")

    content2 = vector.query(question2, EmbeddingModel=embedding, k=1)[0]
    print(f"検索された関連コンテンツ: {content2[:200]}...")
    print(f"\n智谱AI GLM 回答:")
    print(chat.chat(question2, [], content2, lang='ja'))


if __name__ == '__main__':
    main()
