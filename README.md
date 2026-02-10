<p align="center">
  <!-- ロゴ画像をここに配置してください -->
  <!-- <img src="./assets/logo.png" alt="RAG QA System Logo" width="200"> -->
  <h1 align="center">📚 RAG-based QA System</h1>
  <p align="center">
    <strong>ドキュメントを知識に、質問を回答に — RAG で実現する高精度な質問応答システム</strong>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/LLM-ZhipuAI%20GLM--4-blueviolet?style=flat-square" alt="LLM">
    <img src="https://img.shields.io/badge/Framework-RAG-orange?style=flat-square" alt="RAG">
    <img src="https://img.shields.io/badge/Language-中文%20%7C%20日本語-blue?style=flat-square" alt="Language">
  </p>
</p>

---

## 🧠 RAG（検索拡張生成）とは

**RAG（Retrieval-Augmented Generation）** とは、LLM（大規模言語モデル）に **外部知識の検索機能** を組み合わせることで、回答の正確性と信頼性を向上させるアーキテクチャです。

RAG は以下の **3 つのステップ** で動作します：

| ステップ | 処理内容 |
|:---:|:---|
| **🔍 検索（Retrieval）** | ユーザーの質問をもとに、外部の知識ソース（データベース等）から関連情報を取得する |
| **📝 拡張（Augmentation）** | 検索で取得した関連情報をプロンプトに組み合わせ、コンテキストを拡張する |
| **💬 生成（Generation）** | 拡張されたプロンプトを LLM に入力し、根拠に基づいた回答を生成する |

<p align="center">
  <img src="image\ragu_kousei_zu03.png" alt="RAG アーキテクチャ概要図" width="800">
</p>

<p align="center">
  <sub> 図の出典: <a href="https://www.araya.org/projects/rag_kousei_zukai/">株式会社アラヤ — RAGの構成を図解｜非エンジニア向けに仕組みから活用例まで解説</a></sub>
</p>

> 💡 LLM 単体では最新情報や特定ドメインの知識に正確に回答できないことがあります。RAG はこの課題を、**外部ドキュメントの知識を検索・活用** することで解決します。

---

## 📖 プロジェクト概要

**RAG-based QA System** は、上記の RAG アーキテクチャを Python でゼロから実装した質問応答システムです。

ユーザーが提供するドキュメント（PDF / Markdown / テキスト）を読み込み、ベクトルデータベースとして保存し、自然言語の質問に対して **関連するコンテキストを検索** → **LLM が回答を生成** するパイプラインを実現します。

### ✨ 主な機能

- 🔍 **ドキュメントの自動読み込み・チャンク分割** — PDF、Markdown、テキストファイルに対応
- 🧮 **ベクトル化 & 類似度検索** — コサイン類似度による高精度な関連文書検索
- 🤖 **マルチ LLM 対応** — ZhipuAI GLM-4 / OpenAI GPT / InternLM を選択可能
- 🌐 **多言語サポート** — 中国語 🇨🇳 と日本語 🇯🇵 のプロンプトテンプレートを内蔵
- 💾 **ベクトルデータベースの永続化** — JSON 形式で保存・再利用が可能
- ⚡ **ローカル & API 両対応** — API モード（ZhipuAI / OpenAI）とローカルモデル（Jina / InternLM）を柔軟に切替

---

## 🛠️ 技術スタック

| カテゴリ | 技術 |
|:---|:---|
| **言語** | Python 3.11+ |
| **LLM（API）** | [ZhipuAI GLM-4](https://open.bigmodel.cn/)、[OpenAI GPT](https://openai.com/) |
| **LLM（ローカル）** | [InternLM2](https://github.com/InternLM/InternLM)（Transformers） |
| **Embedding（API）** | ZhipuAI Embedding-2、OpenAI text-embedding-3-large |
| **Embedding（ローカル）** | [Jina Embeddings v2](https://huggingface.co/jinaai/jina-embeddings-v2-base-zh) |
| **ベクトル計算** | NumPy |
| **トークナイザー** | tiktoken |
| **ドキュメント処理** | PyPDF2、markdown、html2text |
| **環境管理** | python-dotenv |
| **深層学習基盤** | PyTorch、Transformers |

---

## 📂 プロジェクト構成

```
Project_RAG-based-QA-System/
├── 📄 run.py              # メインスクリプト（中国語版）
├── 📄 run_ja.py           # メインスクリプト（日本語版）
│
├── 📁 RAG/                # コアモジュール
│   ├── LLM.py             #   LLM（大規模言語モデル）ラッパー
│   ├── Embeddings.py      #   テキストベクトル化モジュール
│   ├── VectorBase.py      #   ベクトルデータベース（保存・検索）
│   └── utils.py           #   ファイル読込・チャンク分割ユーティリティ
│
├── 📁 data_ja/            # 日本語サンプルデータ
│   └── rashomon.txt       #   芥川龍之介『羅生門』
│
└── 📁 storage_ja/         # 生成済みベクトルデータベース（日本語）
    ├── doecment.json      #   分割済みドキュメント
    └── vectors.json       #   ベクトルデータ
```

> **Note**: `data/` フォルダ（中国語ドキュメント用）および `storage/` フォルダ（中国語ベクトルデータベース）は、ユーザーがドキュメントを配置し実行することで自動生成されます。

---

## 🚀 クイックスタート

### 前提条件

- **Python** 3.11 以上
- **pip**（パッケージマネージャー）
- **ZhipuAI API キー**（[智谱AI オープンプラットフォーム](https://open.bigmodel.cn/) で取得）
- （任意）**OpenAI API キー** — OpenAI モデルを利用する場合
- （任意）**CUDA 対応 GPU** — ローカルモデル（InternLM / Jina）を利用する場合

### インストール手順

```bash
# 1. リポジトリをクローン
git clone https://github.com/miracle-huang/Project_RAG-based-QA-System.git
cd Project_RAG-based-QA-System

# 2. 仮想環境を作成（推奨）
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

# 3. 依存パッケージをインストール
pip install -r requirements.txt
```

### 環境変数の設定

```bash
# .env.example をコピーして .env を作成
copy .env.example .env    # Windows
# cp .env.example .env    # macOS / Linux
```

`.env` ファイルを編集し、API キーを設定します（詳細は [⚙️ 設定](#️-設定) を参照）。

### 実行方法

#### 🇯🇵 日本語ドキュメントで実行

```bash
python run_ja.py
```

`data_ja/` フォルダ内のドキュメント（芥川龍之介『羅生門』）を読み込み、質問応答を実行します。

**実行例の出力（芥川龍之介『羅生門』を読み込んだ場合）:**

<details>
<summary>📋 クリックして完全な出力を表示</summary>

```
ドキュメントを読み込んでいます...
読み込んだドキュメント数: 8

智谱AI でベクトルを生成しています（数分かかる場合があります）...
Calculating embeddings: 100%|████████████████████████| 8/8 [00:05<00:00,  1.52it/s]

ベクトルデータベースを保存しています...
ベクトルデータベースが storage_ja ディレクトリに保存されました！

══════════════════════════════════════════════════════
 質問 1
══════════════════════════════════════════════════════

質問: 下人は、羅生門の下で、なぜ「行き場のない状態」に追い込まれていたのか。
      本文の内容に基づいて説明せよ。

検索された関連コンテンツ: ある日の暮方の事である。一人の下人が、羅生門の下で
雨やみを待っていた。（中略）当時京都の町は一通りならず衰微していた。今この
下人が、永年、使われていた主人から、暇を出されたのも...

智谱AI GLM 回答:
下人が「行き場のない状態」に追い込まれていた理由は、主に以下の点にあります：

1. **主人からの解雇**: 下人は永年仕えていた主人から四五日前に暇を出された。
   これは当時の京都の衰微という社会的背景による影響である。

2. **京都の荒廃**: 二三年の間に、京都では地震・辻風・火事・饑饉といった
   災いが続き、洛中は著しく衰退していた。

3. **生活の見通しがない**: 雨が止んでも行く当てがなく、「明日の暮しを
   どうにかしようとして、どうにもならない事を、どうにかしようとして」
   途方に暮れていた。

つまり、社会全体の衰退と個人的な失職が重なり、
下人は文字通り「行き所がなくて、途方にくれていた」状態に陥っていたのです。

══════════════════════════════════════════════════════
 質問 2
══════════════════════════════════════════════════════

質問: 老婆が「死人の髪の毛を抜く」行為を正当化した理由は何か。また、
      その説明が、下人のその後の行動にどのような影響を与えたのかを、
      本文に即して述べよ。

検索された関連コンテンツ: 成程な、死人の髪の毛を抜くと云う事は、何ぼう悪い
事かも知れぬ。じゃが、ここにいる死人どもは、皆、そのくらいな事を、されても
いい人間ばかりだぞよ...

智谱AI GLM 回答:
老婆は自らの行為を以下のように正当化しました：

1. **死人たちの生前の行い**: 髪を抜かれている女は、蛇を干魚と偽って
   売っていた人物であり、「されてもいい人間」だと主張した。

2. **生存のための必然性**: 「せねば、饑死をするのじゃて、仕方がなくした事」
   という論理で、生きるためにはやむを得ない行為だと説明した。

この説明が下人に与えた影響は決定的でした：
- 老婆の話を聞く中で、下人の心に「ある勇気」が生まれた
- それは門の下で欠けていた勇気、すなわち「盗人になる」決意であった
- 下人は「では、己が引剥をしようと恨むまいな」と言い放ち、
  老婆の着物を剥ぎ取って夜の闇へと消えた

老婆の「仕方がない」という生存の論理が、皮肉にも下人自身の悪への
転換を正当化する根拠となったのです。
```

</details>

#### 🇨🇳 中国語ドキュメントで実行

```bash
# data/ フォルダにドキュメントを配置してから実行
python run.py
```

> **⚠️ 注意**: `run.py` を実行する前に、プロジェクトルートに `data/` フォルダを作成し、中国語のドキュメント（`.txt` / `.pdf` / `.md`）を配置してください。

#### 📄 独自ドキュメントの利用

1. `data/`（中国語）または `data_ja/`（日本語）フォルダにドキュメントを配置
2. 対応するスクリプト内の `question` 変数を編集して自分の質問に変更
3. スクリプトを実行

---

## ⚙️ 設定

### 環境変数（`.env`）

| 変数名 | 説明 | 必須 |
|:---|:---|:---:|
| `ZHIPUAI_API_KEY` | 智谱AI の API キー | ✅ |
| `OPENAI_API_KEY` | OpenAI の API キー | ❌（OpenAI モデル使用時のみ） |
| `OPENAI_BASE_URL` | OpenAI API のベース URL | ❌（デフォルト: `https://api.openai.com/v1`） |

**`.env.example` テンプレート:**

```env
OPENAI_API_KEY='your openai key'
OPENAI_BASE_URL='https://api.openai.com/v1'

ZHIPUAI_API_KEY='your zhipuai key'
```

> **Caution**: `.env` ファイルには API キーなどの秘密情報が含まれます。**絶対に Git にコミットしないでください**（`.gitignore` に登録済み）。

### LLM / Embedding モデルの切替

`run.py` または `run_ja.py` 内のコードを編集することで、使用するモデルを切替できます。

| モード | Embedding | LLM | 必要環境 |
|:---|:---|:---|:---|
| **API（推奨）** | `ZhipuEmbedding` | `ZhipuChat(model='glm-4-flash')` | API キーのみ |
| **API（OpenAI）** | `OpenAIEmbedding` | `OpenAIChat(model='gpt-3.5-turbo-1106')` | API キーのみ |
| **ローカル** | `JinaEmbedding` | `InternLMChat(path='...')` | GPU + モデルファイル |

---

## 🏗️ アーキテクチャ

```
┌─────────────────────────────────────────────────────┐
│                   RAG パイプライン                     │
│                                                     │
│  📄 ドキュメント                                      │
│    ↓  ReadFiles (utils.py)                          │
│  📝 チャンク分割（トークン制限 + オーバーラップ）          │
│    ↓  Embedding (Embeddings.py)                     │
│  🧮 ベクトル化                                       │
│    ↓  VectorStore (VectorBase.py)                   │
│  💾 ベクトルDB保存（JSON）                             │
│                                                     │
│  ❓ ユーザーの質問                                     │
│    ↓  query() コサイン類似度検索                       │
│  🔍 関連コンテキスト取得                               │
│    ↓  LLM (LLM.py)                                 │
│  💬 回答生成                                          │
└─────────────────────────────────────────────────────┘
```

## 📝 ライセンス

このプロジェクトは [MIT License](LICENSE) の下で公開されています。

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/miracle-huang">miracle-huang</a>
</p>
