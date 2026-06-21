# Purikura Test

FastAPI + OpenCV + MediaPipe で作る、リアルタイムなプリクラ風カメラ加工プロトタイプです。

Mac内蔵カメラまたはOpenCVで扱えるUSBカメラの映像に対して、顔ランドマーク検出、人物セグメンテーション、肌補正、目拡大、小顔補正、メイク風トーン、PNGフレーム合成を行い、撮影画像をSQLiteへ保存します。表示はブラウザUIなので、Mac画面だけでなくHDMI接続したテレビにも同じ画面を表示できます。

## 主な機能

- ブラウザでのリアルタイムMJPEGプレビュー
- Mac内蔵カメラとUSBカメラへ拡張しやすいカメラ抽象化
- MediaPipe Face Landmarker による顔ランドマーク検出
- MediaPipe SelfieMulticlass Image Segmenter による `background / hair / body-skin / face-skin / clothes` 領域抽出
- 肌平滑化、美白、目拡大、小顔、目のハイライト、リップ、チーク、全体トーン補正
- 参考画像寄せの「ドール盛り」補正: 陶器肌、丸い目、アイライン、まつ毛、涙袋、瞳グロス、頬グラデーション、リップグロス、髪なめらか化、白背景化、ソフトグロー
- 透過PNGフレームのアップロードと合成
- 撮影画像と加工設定のSQLite保存
- `Quality` と `Fast` の2種類の処理プロファイル
- デバッグオーバーレイによるランドマーク、マスク、顔パーツ領域の確認
- `/api/performance` による直近の処理時間、FPS、ドロップフレーム数の確認

## 処理プロファイル

このアプリには、画質優先の **標準版（Quality）** と、リアルタイム性優先の **Fast版（Fast）** があります。UIの `Processing profile` から切り替えられます。

### 標準版（Quality）

標準版は、現在の補正品質を優先するプロファイルです。MediaPipeの顔ランドマークとSelfieMulticlassセグメンテーションを使い、顔全体、髪際、耳/首に近い肌、目、眉、鼻、頬、額、顎、唇を細かく分けて処理します。

処理内容:

- 入力フレームを基本的にそのままの解像度で加工します。
- Face Landmarkerで478点ランドマークを取得し、目、眉、鼻、鼻筋、頬、額、顎、唇、顔輪郭を作ります。
- SelfieMulticlassの `face-skin / body-skin / hair` を使い、顔の楕円だけに頼らず、耳や首に近い肌、髪際も含めて補正領域を決めます。
- 目、唇、眉は強い肌ぼかしから保護します。
- 肌領域には強めのbilateral filterとGaussian blurをかけ、美白とピンク寄せを行います。
- 髪際は肌と同じ強度ではなく、顔との境界が浮かない程度に低強度でトーンだけ寄せます。
- 鼻はテカリを抑え、鼻筋には軽いハイライトを加えます。
- 頬にはチーク、唇にはリップ色、目にはキャッチライトを加えます。
- 最後に全体を明るめ、低コントラスト、白/ピンク寄りに調整します。

向いている用途:

- 撮影結果の見た目を優先したい場合
- デバッグオーバーレイでマスクや顔パーツ領域を細かく確認したい場合
- 画質検証や補正ロジックの調整を行う場合

注意点:

- 1280x720相当では処理が重く、環境によってはプレビューFPSが大きく下がります。
- セグメンテーション更新時や全画面の肌/部位別フィルタが主なボトルネックになります。

### Fast版（Fast）

Fast版は、ライブプレビューの遅延とフレームレートを優先するプロファイルです。標準版の考え方は維持しつつ、処理解像度、推論頻度、補正対象領域、画像処理回数を絞っています。

処理内容:

- 入力フレームを内部的に横幅640pxへ縮小して加工し、最後に元解像度へ戻します。
- フレームPNG合成は最終解像度で行うため、アップロードしたフレームの表示サイズは維持されます。
- Face Landmarkerは2フレームごとに実行し、間のフレームは直近の顔検出結果を使います。
- SelfieMulticlass Segmenterは8フレームごとに実行し、間のフレームは前回マスクをEMAで滑らかに使います。
- 肌、頬、額、顎、鼻などの補正は、全画面ではなく `head_roi` に限定します。
- bilateral filterとblurは共有の1回にまとめ、頬/額/顎ごとに重いフィルタを再実行しません。
- 通常時はデバッグ描画を完全にスキップし、`Debug overlay` が有効なときだけ描画します。
- MJPEGプレビュー用JPEGは加工スレッドで一度だけ生成し、配信時に毎回再encodeしないようにしています。
- カメラ読み取りと加工処理を分離し、加工が追いつかない場合は古いフレームを捨てて最新フレームを優先します。

向いている用途:

- 実際の撮影画面として滑らかなプレビューを出したい場合
- HDMI接続したテレビなど、大きい画面へ低遅延で表示したい場合
- カメラ位置やポーズ調整中の操作感を優先したい場合

注意点:

- 内部処理解像度を下げるため、標準版より細部の補正精度は落ちます。
- 推論を間引くため、急に顔が大きく動いた瞬間はマスクや顔パーツ領域が少し遅れて追従することがあります。
- デバッグオーバーレイを有効にすると描画処理が増えるため、Fast版でもFPSは下がります。

## プロファイルの使い分け

| 項目 | 標準版（Quality） | Fast版（Fast） |
| --- | --- | --- |
| 優先するもの | 補正品質 | 低遅延、FPS |
| 内部処理解像度 | 入力解像度中心 | 横幅640px |
| 顔検出 | 毎フレーム | 2フレームごと |
| セグメンテーション | 4フレームごと | 8フレームごと |
| 補正範囲 | 広め、部位別に丁寧 | head ROI中心 |
| 重いフィルタ | 肌/部位ごとに適用 | 共有して回数削減 |
| おすすめ用途 | 品質確認、撮影結果重視 | ライブプレビュー、実運用 |

基本運用では、プレビュー中は `Fast`、補正の見た目を詰めるときや撮影品質を確認するときは `Quality` を使う想定です。

## エフェクト操作

UIのエフェクト操作は、実際に撮影中に触る項目だけに絞っています。以前の詳細スライダーで分かれていた美白、チーク、リップ、アイライン、まつ毛、涙袋、瞳グロス、髪のなめらかさ、ソフトグローは、`Purikura strength` と `Doll style` から内部で自動配分します。

- `Purikura strength`: 全体の明るさ、美白、ピンク寄せ、低コントラスト化をまとめて調整します。
- `Skin smoothing`: 顔、首、手などの肌領域をなめらかにします。目、眉、唇は保護します。
- `Eye enlarge`: 目周辺を局所変形して大きく見せます。
- `Face slim`: 頬下と顎を内側へ寄せ、Vライン寄りの小顔にします。
- `Doll style`: 陶器肌、丸い目、アイライン、まつ毛、涙袋、瞳グロス、チーク、リップ、髪のなめらかさ、ソフトグローをまとめて強めます。
- `Background`: background maskだけを白/薄ピンクへ寄せ、白いスタジオ背景に近づけます。
- `Debug overlay`: ランドマーク、マスク、部位領域を重ねて表示します。

処理順は、顔/人物解析、形状変形、肌/髪/背景補正、アイメイク、頬/唇、全体グローです。Fast版でも同じ方向性を保ちますが、横幅640pxの処理画像上で軽量近似します。

## デバッグオーバーレイ

UIの `Debug overlay` では、補正対象領域を画面上に重ねて確認できます。

- `Off`: デバッグ描画なし。通常運用向けです。
- `Landmarks`: 顔bbox、顔輪郭、目、唇、主要ランドマークを表示します。
- `Masks`: background、head、skin、hair、clothes、protected maskを色付き半透明で表示します。
- `Parts`: 鼻、額、顎、頬、眉などの部位領域を表示します。
- `All`: 上記をまとめて表示します。

領域選定のズレ、肌補正の抜け、髪際の不自然さ、目/唇/眉の保護状態を確認するときに使います。

## 起動方法

```bash
uv sync --all-extras
uv run python scripts/download_models.py
uv run uvicorn purikura_test.app:app --reload
```

ブラウザで <http://127.0.0.1:8000> を開きます。

macOSでは、TerminalまたはUvicornを起動しているアプリにカメラアクセス許可を与えてください。

## テスト

```bash
uv run pytest
```

## ベンチマーク

```bash
uv run python scripts/benchmark_pipeline.py --profile both
```

任意の画像で測る場合:

```bash
uv run python scripts/benchmark_pipeline.py --profile both --image /path/to/image.png
```

出力例:

```text
quality: avg=438.5ms p95=601.4ms min=379.7ms max=601.4ms fps=2.3
fast: avg=56.9ms p95=193.0ms min=32.6ms max=193.0ms fps=17.6
```

この値は実行環境、入力画像、顔の大きさ、Debug overlayの有無で変わります。Fast版の目標は、Mac内蔵カメラ相当の入力で平均15fps以上です。

## APIメモ

- `GET /api/effects`: 現在の補正設定を返します。
- `PUT /api/effects`: `processing_profile` を含む補正設定を更新します。
- `GET /api/performance`: 直近の `processing_ms`, `encode_ms`, `effective_fps`, `dropped_frames`, `profile` を返します。
- `GET /api/preview.mjpeg`: 加工済みライブプレビューをMJPEGで返します。
- `POST /api/captures`: 現在の加工済みフレームをJPEGとしてDBへ保存します。

`processing_profile` は `"quality"` または `"fast"` を指定します。撮影時の `effect_settings_json` にも保存されるため、後からどちらのプロファイルで撮影されたか追跡できます。

公開している補正設定は `processing_profile`, `purikura_intensity`, `skin_smoothing`, `eye_enlarge`, `face_slim`, `doll_intensity`, `background_high_key`, `debug_overlay` です。削除済みの詳細パラメータを送った場合は422になります。

## モデルと保存先

- `scripts/download_models.py` は `models/face_landmarker.task` と `models/selfie_multiclass_256x256.tflite` を取得します。
- モデルファイルはgit管理しません。
- 撮影画像はJPEG blobとして `data/purikura.sqlite3` に保存されます。
- アップロードしたPNGフレームもDBに保存され、プレビュー/撮影時に現在のフレーム解像度へリサイズして合成されます。

## カメラと外部表示

- 初期カメラはOpenCV camera indexを使います。Mac内蔵カメラは通常 `0`、USBカメラは追加indexとして見えます。
- カメラ実装は `CameraSource` 抽象に分けているため、将来USBカメラ固有の制御が必要になっても差し替えやすい構成です。
- 表示はブラウザUIなので、HDMI接続したテレビへはOS側で外部ディスプレイ表示またはミラーリングを行い、同じWeb UIを表示します。

## 品質版の保全

高速化前の品質版はgit tag `quality-baseline-85f4d5a` として残しています。Fast版の調整で見た目が崩れた場合は、このtagと比較して標準版の挙動を確認できます。
