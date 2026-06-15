# LLMオンボーディングサマリー

> このドキュメントは、新任LLMエージェントが Triangle Splatting の現在の作業状態を短時間で把握し、同じ環境で検証を継続するための初期資料です。

## 1. プロジェクト概要と目的

- **プロジェクト名称・領域:** Triangle Splatting for Real-Time Radiance Field Rendering
- **最終成果物:** Blackwell GPU / CUDA 13.0 ドライバ環境上で、pixi による再現可能な学習・レンダリング検証環境を整備する。
- **ビジネス背景・価値:** 3D Gaussian Splatting 系の派生手法を、手元の RTX PRO 4000 Blackwell 環境で比較・検証できる状態にする。
- **現時点の進捗サマリ:** `work/pixi-pr7-pr47` ブランチで upstream PR #7 と PR #47 を取り込み、pixi 環境・CUDA extension build・501枚データの smoke 学習まで確認済み。

## 2. クリティカルな要求・制約

> 「壊してはいけない」品質・仕様ラインを箇条書きで列挙します。

- origin への push はユーザー指示がある場合のみ行う。今回は commit & push 指示あり。
- 生成データ、学習出力、ログ、pixi 実体環境は commit しない。`.gitignore` の `data/`, `outputs/`, `logs/`, `.pixi/` を守る。
- `simple-knn` の元 GitLab submodule URL は取得不能だったため、GitHub mirror `https://github.com/camenduru/simple-knn.git` を使う。
- PR #47 の `diff-triangle-rasterization` submodule pointer `62bbc103...` は upstream から fetch 不能だったため、公式 `6d61f4c...` に戻し、build 時だけ `#include <cstdint>` workaround を適用する。
- Blackwell GPU では PyTorch `2.8.0+cu128` と pixi の CUDA 12.8 nvcc を使う。ドライバ表示は CUDA 13.0 でも、toolchain は CUDA 12.8 で固定している。

## 3. 参照すべき合意済み資料

> 新任エージェントが必ず確認すべき一次資料の一覧です。パスと役割を記載します。

| 種別 | ファイル/リンク | 概要・用途 |
|------|------------------|------------|
| README | `README.md` | upstream の基本手順と、追加した pixi 手順 |
| 環境定義 | `pixi.toml` / `pixi.lock` | Python 3.11、CUDA 12.8、PyTorch cu128、pip 依存の固定 |
| CUDA 環境設定 | `scripts/pixi-cuda-env.sh` | `CUDA_HOME`, include/lib path, `TORCH_CUDA_ARCH_LIST=12.0+PTX` の設定 |
| extension build | `scripts/install_diff_triangle.sh` | `diff-triangle-rasterization` の一時 cstdint patch と pip install |
| 検証スクリプト | `scripts/verify_environment.py` | package、CUDA、nvcc、extension import の確認 |
| 取り込みPR | https://github.com/trianglesplatting/triangle-splatting/pull/7 | `create_ply.py` の追加 |
| 取り込みPR | https://github.com/trianglesplatting/triangle-splatting/pull/47 | CUDA build fix 系の提案。submodule pointer はそのまま再現不可 |
| 既知課題 | https://github.com/graphdeco-inria/gaussian-splatting/issues/265 | `simple-knn` GitLab URL 取得不能に関する関連 issue |

## 4. タスク境界（任せること / 任せないこと）

### 任せるタスク

- pixi 環境の再構築、`pixi run verify` / `pixi run smoke` の再実行。
- 501枚データを使った短時間 smoke 学習と、保存モデルの読み戻し確認。
- 生成物を commit 対象から除外しつつ、必要な scripts/docs/config のみを commit する。

### 任せないタスク

- ユーザー未承認の長時間本学習や大量出力の生成。
- `data/`, `outputs/`, `logs/`, `.pixi/` の commit。
- submodule の取得不能 commit を無理に pointer として固定すること。
- ユーザー指示なしの force push、履歴改変、main への直接作業。

## 5. インタラクション方針

- **回答スタイル:** 日本語で簡潔に、実行したコマンド・結果・パスを明記する。
- **回答手順:** 前提、実施内容、検証結果、残タスクの順で報告する。
- **禁止事項・注意:** 未確認の依存バージョンや外部PRの状態を断定しない。必要なら GitHub / local git で確認する。
- **秘匿情報の扱い:** GitHub token、ローカル認証情報、ユーザー固有の秘密情報は出力しない。

## 6. 試行タスク（オンボーディング演習）

> 小さな検証タスクを2〜3件記載してください。理解度を確認するために実施します。

1. `pixi run verify` を実行し、PyTorch `2.8.0+cu128`、GPU `NVIDIA RTX PRO 4000 Blackwell`、nvcc CUDA 12.8 が見えることを確認する。
2. `pixi run smoke` を実行し、`diff_triangle_rasterization` と `simple_knn._C` の import が成功することを確認する。
3. 501枚データの smoke 入力を `data/tva_0501_gluemap_aba_smoke` に作り、20 iterations の学習が保存まで通ることを確認する。

## 7. 運用ルール・変更管理

- **ドキュメント更新時の記載ルール:** 実行したコマンド、出力先、検証結果、既知の回避策を具体的に書く。
- **TBDの扱い:** 未確認事項は TBD として残し、推測で完了扱いにしない。
- **レビュー/承認フロー:** push や長時間学習はユーザー指示を待つ。PR 作成は別途指示がある場合に行う。
- **その他の運用ルール:** 生成物は `.gitignore` に入れ、必要な小さな再現スクリプトや docs だけを version 管理する。

---

### 付録: 参考情報

- **主要リポジトリ/ディレクトリ:** `/home/kasm-user/Desktop/triangle-splatting`
- **作業ブランチ:** `work/pixi-pr7-pr47`
- **代表的なコマンド:**

```bash
pixi install
pixi run verify
pixi run install-extensions
pixi run smoke
```

- **501枚 smoke 学習コマンド:**

```bash
pixi run python train.py \
  -s data/tva_0501_gluemap_aba_smoke \
  -m outputs/tva_0501_gluemap_aba_smoke_YYYYMMDDTHHMMSSZ \
  --iterations 20 \
  --resolution 4 \
  --test_iterations -1 \
  --save_iterations -1
```

- **確認済み smoke 結果:** 2026-06-15 に 20 iterations が成功。保存モデルは `triangles_points (52500, 3, 3)` として読み戻し確認済み。
- **依存ライブラリ:** PyTorch `2.8.0+cu128`, torchvision `0.23.0+cu128`, numpy `1.26.4`, Open3D `0.18.0`, lpips `0.1.4`, mediapy `1.2.6`, opencv-python `4.11.0`。
- **連絡先/責任者:** TBD
