# リアルタイム追従性・高速化ロードマップ

## 目的

現状の課題は、顔の動きに対して加工が追いつかず、肌補正や背景補正が前の顔位置に残って見えることです。これは単純なFPS不足だけではなく、以下が同時に起きることで発生します。

- 重い加工フレームが完了した時点で、すでに顔が別位置へ移動している。
- 顔ランドマークやセグメンテーションmaskを間引き・キャッシュしているため、解析結果が映像より古い。
- maskのEMAが画素位置固定で混ざると、前フレームの顔位置が薄く残る。
- MJPEG配信ではブラウザ側のバッファやバックプレッシャを細かく制御しにくい。
- 高品質なプリクラ加工は、局所変形、肌/髪/背景mask、複数blend、blur/glowが多く、CPUメモリ帯域を使う。

このドキュメントでは、今後取り得る高速化と残像対策を網羅的に整理します。方針は、プレビューは低遅延、撮影保存は高品質に分けることです。フォールバックは不要という前提に合わせ、必須モデルや必須処理が失敗した場合は明示的に止めます。

## 参考資料

- [MediaPipe Face Landmarker Python](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python)
  - `VIDEO` / `LIVE_STREAM` mode、`detect_for_video` / `detect_async`、trackingによる遅延削減。
- [MediaPipe Image Segmenter Python](https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter/python)
  - `segment_for_video` / `segment_async`、category/confidence mask。
- [MediaPipe Real-time Streams](https://ai.google.dev/edge/mediapipe/framework/framework_concepts/realtime_streams)
  - timestamp、timestamp bounds、リアルタイムgraphで下流を待たせない考え方。
- [OpenCV Performance Measurement and Improvement Techniques](https://docs.opencv.org/4.x/dc/d71/tutorial_py_optimization.html)
  - `cv.useOptimized()` / `cv.setUseOptimized()`、計測優先、SIMD最適化。
- [Apple Core Image Performance Best Practices](https://developer.apple.com/library/archive/documentation/GraphicsImaging/Conceptual/CoreImaging/ci_performance/ci_performance.html)
  - `CIContext`再利用、小さい画像、CPU/GPU転送削減、色管理コスト。
- [Apple Metal](https://developer.apple.com/metal/)
  - Apple silicon上のGPU compute、profiling、Metal Performance Shaders。
- [LiteRT GPU delegate](https://developers.google.com/edge/litert/performance/gpu)
  - GPU推論の低遅延化と、対応op/転送コストの注意点。
- [ONNX Runtime CoreML Execution Provider](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
  - macOS/Core ML経由の推論、対応op、CoreML provider登録。

## 現状実装の整理

対象コード:

- `src/purikura_test/runtime.py`
- `src/purikura_test/effects.py`
- `src/purikura_test/api_models.py`
- `scripts/benchmark_pipeline.py`

現状は `quality` と `fast` の2プロファイルがあります。

- `quality`
  - 画質優先。
  - MediaPipe Face LandmarkerとSelfieMulticlass Segmenterを使い、肌、髪、背景、目、唇、頬、鼻、輪郭、glowを強めに処理する。
  - previewで使うと重くなりやすい。
- `fast`
  - preview優先。
  - 横幅640pxへ縮小して処理し、元解像度へ戻す。
  - Face Landmarkerは毎フレーム、Segmenterは原則8フレームごと。
  - 顔移動が大きい場合、mask EMAを止め、新しいmaskを使う。
  - 顔移動が小さい場合、前回maskを顔中心移動分だけ平行移動して再利用/EMAする。
  - `motion_factor` が高い間は、肌、変形、メイク、背景、glowを弱める。

すでに入っている残像対策:

- `FramePacket(id, captured_at, frame)` を付与。
- 加工完了時に `latest_raw_frame_id - packet.id` が閾値を超えていたらpublishしない。
- Fastは1フレーム超過、Qualityは2フレーム超過を破棄対象にする。
- `/api/performance` に `discarded_processed_frames`, `frame_age_ms`, `landmark_age_ms`, `mask_age_ms`, `motion_factor`, `publish_interval_ms`, `publish_lag_frames`, `preview_stall_ms`, `published_frame_id`, `latest_raw_frame_id` を返す。
- Face Landmarkerの検出0件時キャッシュTTLは100ms。
- Fast previewでは顔ランドマークを毎フレーム更新している。
- Fast previewではmotionに応じてエフェクト強度を減衰する。
- SelfieMulticlass Segmenterは固定間引きだけでなく、顔移動量と `mask_age_ms` を見て前倒し更新する。静止時はFastで最大8フレーム再利用し、motion中またはmask age 150ms超過時は古いmaskを使い続けない。
- 撮影保存では最新rawフレームを固定し、`processing_profile="quality"` にした設定で再処理してDBへ保存する。
- 起動直後のpreviewは `processing_profile="fast"` を初期値にして、最初の表示が重いQuality処理で詰まらないようにしている。
- `/api/cameras` は現在使用中のカメラを再Openせず、macOS/OpenCVで起動中キャプチャを不安定にしないようにしている。
- Mac内蔵カメラ向けにread loopを約30fpsへ抑制し、preview publish判定はフレームIDだけでなく経過時間も併用している。一定時間publishが止まった場合は次の加工フレームを通し、静止画化を避ける。
- UIのPerformanceパネルで処理時間、FPS、表示フレーム年齢、landmark/mask年齢、publish間隔、破棄数を確認できる。

残る課題:

- すでに加工中の重いフレームは最後まで計算されるため、CPU時間は消費する。
- Segmenter maskの再利用は平行移動のみで、再セグメントまでの短い間は回転、スケール、顔向き変化には弱い。
- ブラウザ側ではframe idを見て古いフレームを捨てられない。
- DB metadataには現在、撮影時Quality設定は残るが、preview時のprofileは別フィールドとしては残していない。

## 重要な判断基準

残像問題では、平均FPSよりも「表示しているフレームがどれだけ新しいか」を優先します。

目標値:

| 指標 | 目標 | 理由 |
| --- | ---: | --- |
| `frame_age_ms` | 100ms以下 | 体感上の追従性を保つ |
| `landmark_age_ms` | 66ms以下 | 15fps相当で目/輪郭変形のズレを抑える |
| `mask_age_ms` | 150ms以下 | 肌/髪/背景maskの残像を抑える |
| Fast処理時間 | 平均70ms以下 | 15fps以上を狙う |
| publish遅延 | 120ms超は破棄 | 古い加工結果を表示しない |
| capture処理 | 数百ms許容 | 撮影保存は高品質優先 |

基本ルール:

- 古い解析結果を無理に使って画質を保つより、動いている間は補正を弱める。
- 最新フレームを表示するために、古いフレームは捨てる。
- previewとcaptureは別物として設計する。
- GPU/ONNX/native化は、計測でボトルネックが明確になってから入れる。

## 高速化候補一覧

| 領域 | 施策 | 残像への効き | 速度への効き | 実装コスト | 優先度 |
| --- | --- | --- | --- | --- | --- |
| 計測 | `landmark_age_ms` / `mask_age_ms`追加 | 高 | 中 | 低 | 最高 |
| publish制御 | 古い処理結果をさらに厳格に破棄 | 高 | 低 | 低 | 最高 |
| preview/capture分離 | previewはFast、撮影時だけQuality再処理 | 高 | 高 | 中 | 最高 |
| mask制御 | motion時にmask EMAを即リセット | 高 | 中 | 低 | 高 |
| mask追従 | landmark affine warp | 高 | 中 | 中 | 高 |
| pipeline | Capture/Analyze/Render/Publish分離 | 高 | 高 | 中 | 高 |
| MediaPipe | `LIVE_STREAM` mode移行 | 高 | 高 | 中 | 高 |
| 出力 | WebSocket/WebRTCで古いframe id破棄 | 中 | 中 | 中 | 中 |
| 画像処理 | ROI化、blend集約、copy削減 | 中 | 高 | 中 | 高 |
| 解像度 | 動的process_width | 中 | 高 | 低 | 高 |
| アルゴリズム | 動き中はlandmark-only mask | 高 | 高 | 中 | 高 |
| native | Rust/PyO3 or C++/pybind11 | 低から中 | 高 | 高 | 中 |
| GPU | Core Image / Metal | 中 | 高 | 高 | 中 |
| 推論 | ONNX/CoreML/LiteRT GPU | 中 | 条件付き | 高 | 低から中 |
| UX | 静止Ready表示、撮影時Quality | 中 | 中 | 低 | 高 |

## Phase 0: 計測を先に増やす

高速化は、まず「古い何が残っているのか」を画面とAPIで見えるようにします。

追加するメトリクス:

- `capture_timestamp_ms`: カメラread時刻。
- `analysis_timestamp_ms`: landmark/segmenterが見たフレーム時刻。
- `render_started_ms`: render開始時刻。
- `render_finished_ms`: render終了時刻。
- `published_timestamp_ms`: `_latest_jpeg` 更新時刻。
- `frame_age_ms`: publish時点でのraw frame年齢。
- `landmark_age_ms`: publish時点でのlandmark年齢。
- `mask_age_ms`: publish時点でのsegment mask年齢。
- `motion_px`: 顔中心の移動量。
- `motion_ratio`: `motion_px / face_width`。
- `inflight_age_ms`: 加工開始から完了までの経過。
- `discarded_inflight_frames`: 加工完了したが古すぎて捨てた数。
- stage別時間:
  - camera read
  - resize
  - face landmarker
  - segmenter
  - mask warp/EMA
  - face warp
  - skin/hair/background
  - makeup
  - glow
  - frame composite
  - JPEG encode

実装箇所:

- `PerformanceSummary` に項目追加。
- `FrameAnalysis` に `source_frame_id` と `source_timestamp` を追加。
- Debug overlayに `frame_age`, `landmark_age`, `mask_age`, `motion` を表示。
- `scripts/benchmark_pipeline.py` にstage別計測を追加。

この段階の狙い:

- FPSではなく残像の根本原因を追える。
- 「推論が遅い」のか「画像処理が遅い」のか「ブラウザが遅れている」のかを切り分ける。

## Phase 1: 残像を直接消す短期施策

### 1. 古い加工結果をpublishしない条件を厳しくする

現状はFastで1フレーム超過まで許容しています。顔移動が大きい場合は0フレーム超過にします。

案:

- `motion_factor < 0.04`: 1フレーム超過まで許容。
- `0.04 <= motion_factor < 0.10`: 0フレーム超過のみ許容。
- `motion_factor >= 0.10`: 処理完了時点で最新rawでなければ破棄。

効果:

- 動きが大きい瞬間に、古い顔位置の加工結果が出にくくなる。

副作用:

- 動いている間はpublishされるフレーム数が減る。
- ただし残像よりフレーム落ちの方が自然に見える。

### 2. 動き中はセグメンテーション由来の補正をさらに弱める

mask残像の主因は、古いskin/hair/background maskです。動き中はlandmark由来の顔近傍だけに寄せます。

案:

- `motion_factor >= 0.08`:
  - `background_high_key = 0`
  - `hair_silk *= 0.3`
  - `soft_glow *= 0.3`
  - `skin_smoothing *= 0.5`
  - `doll_intensity *= 0.6`
- `motion_factor >= 0.14`:
  - segmentation maskを使わず、顔楕円ベースの簡易maskのみ。
  - 目/輪郭変形は最小限。

効果:

- 古い背景白寄せや肌maskが顔の後ろに残りにくい。

### 3. previewとcaptureを完全分離する

previewは追従性最優先、captureは画質最優先にします。

案:

- `PurikuraRuntime` が最新raw frameを保持する。
- previewは常にFastで処理して表示。
- 撮影ボタン押下時に最新rawを固定。
- そのrawをQuality設定で再処理してDBへ保存。
- 保存中はUIに `Saving...` を出す。
- DBの `effect_settings_json` には `capture_profile=quality` と `preview_profile=fast` を残す。

効果:

- previewを軽くしても保存画質を落とさずに済む。
- 動いている間のpreviewは薄くしても、撮影結果はしっかり盛れる。

### 4. 動的process_width

Fastは横幅640px固定ですが、処理が詰まる場面では自動で下げます。

案:

- 通常: 640px。
- `processing_ms > 80` または `frame_age_ms > 120`: 512px。
- `processing_ms > 120` または `discarded_processed_frames` が連続: 448px。
- 1秒以上安定したら段階的に640pxへ戻す。

注意:

- 急に解像度が上下すると見た目が揺れるため、変更は0.5から1秒単位で行う。
- captureはQuality再処理にするのでpreview解像度低下の影響を保存に出さない。

### 5. Fast時はbackground/glowを低頻度化する

背景白寄せとsoft glowは顔追従に必須ではありません。残像が出る時は止める対象です。

案:

- background mask更新はsegmenter更新時のみ。
- glow layerは2から4フレームごとに低解像度で作る。
- motion中はglowを弱めるか完全停止。

効果:

- 全画面処理と残像を同時に減らせる。

## Phase 2: 解析と描画のパイプライン分離

現状はcamera threadとprocessor threadが中心です。次は4段に分けます。

```text
Capture -> Analyze -> Render -> Publish
```

### Capture

- カメラから最新raw frameだけを保持する。
- キューはサイズ1。
- 古いrawは即捨てる。
- `frame_id` と `captured_at` を必ず付ける。

### Analyze

- Face LandmarkerとSegmenterを担当する。
- 最新rawだけを解析対象にする。
- 処理中に新しいrawが来たら、完了後に最新rawへ飛ぶ。
- 結果には `source_frame_id`, `source_timestamp`, `analysis_finished_at` を持たせる。

### Render

- 最新rawと最新analysisを使う。
- analysisが古い場合はage gatingで補正を弱める。
- render完了時点でrawが古くなっていたらpublishしない。

### Publish

- 最新renderだけを保持する。
- JPEG/WebSocket/WebRTC出力に変換する。
- 配信先が詰まってもrender側を待たせない。

この構成の利点:

- 推論が遅くてもカメラ取得が止まらない。
- renderが遅くてもanalyzeが止まらない。
- 古い解析結果を使うかどうかをtimestampで明示判断できる。

## Phase 3: MediaPipe `LIVE_STREAM` mode移行

MediaPipeのPython Tasksは、`VIDEO` modeでは呼び出しスレッドをブロックします。`LIVE_STREAM` modeでは `detect_async` / `segment_async` が即時returnし、callbackで結果を受け取ります。Face Landmarkerでは、処理中に新しい入力が来た場合にその入力を無視する仕様があり、低遅延設計と相性があります。

設計案:

- `AsyncFaceTracker`
  - `detect_async(mp_image, timestamp_ms)` を最新フレームだけ投入。
  - callbackで `FaceAnalysis(frame_id, timestamp, detections)` を保存。
  - 未完了中は追加投入しすぎない。
- `AsyncHeadSegmenter`
  - `segment_async(mp_image, timestamp_ms)` をmotion/age条件に応じて投入。
  - callbackで `SegmentationAnalysis(frame_id, timestamp, masks)` を保存。
- renderer
  - 最新rawに対して、analysisのageとsource_frame_idを評価。
  - 古いanalysisなら補正を薄めるか使わない。

注意点:

- callbackは別スレッドで来る前提でロック設計が必要。
- 全フレームの結果が返るとは限らない。
- timestampは単調増加が必須。
- result callback内で重い処理をしない。mask変換やresizeは別スレッドへ渡す。

優先度:

- 高。Pythonのまま残像を減らす次の大きな打ち手。

## Phase 4: mask追従アルゴリズム

### 1. affine warp

今は前回maskを顔中心の平行移動で合わせています。次は回転、拡大縮小、顔向き変化へ対応します。

代表点:

- 左目中心
- 右目中心
- 鼻先
- 口中心
- 顎
- 左右頬

手順:

1. 前回ランドマーク代表点と今回ランドマーク代表点を対応させる。
2. `cv2.estimateAffinePartial2D` でaffineを推定する。
3. 前回 `head/skin/hair/background/protected` maskを `cv2.warpAffine` する。
4. 新しいsegmentationが来た場合だけEMAで混ぜる。
5. motionが大きすぎる、または推定失敗時はmaskを捨てる。

効果:

- 横移動だけでなく、顔の傾きや距離変化でもmask残像が減る。

### 2. piecewise affine / triangle warp

目、鼻、口、輪郭など複数三角形で局所的にmaskを動かします。

効果:

- 顔の表情や口の開閉により強い。

副作用:

- 実装コストが高い。
- mask境界が破綻しやすいので、まずaffineで十分か確認する。

### 3. optical flow

Lucas-Kanade optical flowやdense optical flowで顔ROI内の動きを推定します。

候補:

- sparse LK flowで代表点のmedian移動を取る。
- dense Farnebackは重いためFast previewには慎重。
- flow信頼度が低い場合はmaskを破棄する。

優先順位:

- landmark affineの後。
- ランドマークが不安定な照明・角度でだけ検討。

### 4. landmark-only mask

高速移動中はSelfieMulticlassを追いかけず、顔ランドマークから作る簡易maskだけを使います。

対象:

- 顔楕円を少し拡張。
- 目/眉/唇をprotect。
- 頬/鼻/額/顎はランドマーク領域で補正。
- 背景、髪、服は補正しない。

効果:

- mask残像がほぼ消える。
- 動き中の画質は落ちるが、静止時に高品質へ戻せる。

## Phase 5: Python/OpenCV内の画像処理削減

### 1. ROI処理の徹底

今後さらに見るべき箇所:

- `apply_hair_silk`
- `apply_background_high_key`
- `apply_soft_glow`
- `apply_doll_makeup`
- `blend_by_mask`
- `alpha_composite_bgra`

方針:

- head/face/hand ROIだけでskin/hair/makeupを処理。
- backgroundは低解像度maskで計算し、最終合成だけ全画面。
- 服は基本保護し、処理対象から外す。

### 2. blend集約

現在は処理ごとに `blend_by_mask` を呼び、array copyが増えます。

案:

- 各処理で `target_layer` と `alpha_mask` を作る。
- 最後に1回または2回でまとめて合成。
- `float32` 変換を関数ごとに繰り返さない。
- `np.ascontiguousarray` を必要箇所に限定する。

期待効果:

- CPU時間よりもメモリ帯域の削減が効く可能性が高い。

### 3. bilateral filterの削減

bilateralは肌補正には効きますが重い処理です。

案:

- Fast版では `GaussianBlur + mask保護 + tone補正` へ寄せる。
- Qualityのみbilateralを維持。
- 頬、額、顎で別々にblurせず、共有blurを1回作る。
- 顔ROI内だけ処理する。

### 4. 色空間変換の共有

Lab/HSV/BGR/RGB変換を複数回行っています。

案:

- 1フレーム内でLab/HSVを共有する。
- tone補正はLUT化する。
- MediaPipe入力用RGB変換とOpenCV処理用BGRを明確に分離する。
- RGB/BGR往復を増やさない。

### 5. OpenCV最適化とスレッド設定

確認項目:

- `cv2.useOptimized()` がTrueか。
- 起動時に `cv2.setUseOptimized(True)` を呼ぶ。
- `cv2.getNumThreads()` を記録する。
- `cv2.setNumThreads(n)` を1, 2, 4, defaultでベンチする。

注意:

- OpenCVのスレッド数を増やすと、MediaPipeやUvicornとCPUを奪い合い、逆に遅くなる場合がある。

## Phase 6: 出力方式の改善

### 1. WebSocket + binary frame

MJPEGをWebSocketに変えると、クライアント側でframe idを見て古いフレームを捨てられます。

設計:

- serverが `{frame_id, captured_at, processed_at, jpeg_blob}` を送る。
- clientは受信時に現在表示中より古いframe idなら捨てる。
- `requestAnimationFrame` で最新blobだけ描画する。

利点:

- ブラウザ内部のMJPEGバッファに依存しない。
- Debug overlayやperformance情報も同じchannelで送れる。

### 2. WebRTC

低遅延映像配信としてはWebRTCが自然です。

利点:

- ブラウザが映像用途のバッファ制御を持つ。
- 将来、HDMIテレビ表示や遠隔表示にも展開しやすい。

欠点:

- 実装が増える。
- Python側の映像エンコード/送出設計が必要。

### 3. WebCodecs

ブラウザ側でフレームを細かく制御できます。

注意:

- 対応ブラウザと実装コストを確認する。
- まずWebSocketで十分な可能性が高い。

## Phase 7: native化

Python/OpenCVでの整理後、まだFastが15fps未満ならnative化を検討します。

### Rust/PyO3

向いている処理:

- mask blend
- multiple mask merge
- EMA
- affine warp前後のmask整形
- tone LUT
- ROI composite
- local warp/zoomの一部

利点:

- メモリ安全性。
- `maturin` と `uv` 管理の相性が良い。
- Python APIを保ちやすい。

注意:

- NumPy arrayをzero-copyで扱う設計が必要。
- OpenCV連携はC++ほど自然ではない。

### C++/pybind11

向いている処理:

- OpenCV C++へまとめて移す処理。
- blur、warp、resize、blend、LUTを1つの関数にまとめる。

利点:

- OpenCV C++との相性が良い。
- Python境界と中間copyを減らせる。

注意:

- build環境が重くなる。
- macOS/Apple siliconのwheel作成を管理する必要がある。

### OpenCV G-API

複数OpenCV処理をgraph化して最適化する選択肢です。

優先度:

- 低から中。
- まずC++/pybind11でhot pathを直接まとめる方が検証しやすい。

## Phase 8: macOS GPU活用

### Core Image

向いている処理:

- blur
- color matrix
- LUT
- blend
- bloom/glow
- background high key

実装方針:

- `CIContext` を毎フレーム作らず再利用。
- 低解像度で処理できるものは小さく処理する。
- CPU/GPU間転送を最小化する。
- 色管理をどこまで必要にするか決める。

注意:

- Pythonから直接扱う場合はPyObjCなどが必要。
- OpenCVのNumPy arrayとCore Imageの変換コストが支配的になる可能性がある。

### Metal

向いている処理:

- mask blend
- local warp
- tone
- glow
- multi-layer composite

利点:

- Apple siliconでは長期的に最も性能を出しやすい。
- 複数処理を1から2個のcompute shaderへまとめられる。

注意:

- 実装コストが高い。
- Pythonアプリから呼ぶならSwift/Objective-C/C++ bridgeが必要。
- GPUへ送って戻すだけの処理では遅くなる可能性がある。

優先度:

- Python/OpenCVとnative化で足りない場合の長期施策。

## Phase 9: 推論基盤の変更

### ONNX/CoreML

検討条件:

- stage別計測でFace LandmarkerまたはSegmenterが支配的になった場合。
- 画像処理の最適化後も推論がボトルネックである場合。

検証項目:

- Face Landmarker相当のモデルをONNX/CoreMLへ変換できるか。
- SelfieMulticlass相当のモデルを変換できるか。
- CoreML Execution Providerの対応opに収まるか。
- 初回compile時間。
- CPU/GPU/ANEの実効latency。
- ランドマーク座標やmask品質の差分。

注意:

- 現状の主因は古い解析結果と画像処理遅延なので、最初の打ち手ではない。

### LiteRT GPU delegate

検討条件:

- TFLiteモデル単体の推論が重いと判明した場合。
- macOS環境でdelegateを安定利用できる見通しがある場合。

注意:

- 非対応opが混じるとCPU/GPU同期で逆に遅くなる可能性がある。
- MediaPipe Tasks API経由でどこまで制御できるか確認が必要。

### MediaPipe Framework graph

Python Tasks APIではなく、MediaPipe graphへ寄せる選択肢です。

利点:

- timestamp boundsやcalculator schedulingを明示できる。
- real-time streamとして自然に設計できる。
- Python境界を減らせる可能性がある。

欠点:

- C++/Bazel/graphの実装コストが高い。
- 現アプリからの移行が大きい。

優先度:

- 長期。Python版で設計が固まった後。

## UI/UX側の対策

### 1. Ready判定

顔が止まっている瞬間を検出し、撮影に向く状態をUIに出します。

案:

- `motion_factor < 0.03` が300ms続いたら `Ready`。
- `motion_factor >= 0.08` なら `動いています`。
- 撮影ボタンは常に押せるが、動いている時は「保存品質が落ちる可能性」を表示する。

### 2. カウントダウン中の制御

- カウントダウン中はFast previewを維持。
- シャッター瞬間のrawを固定。
- 固定rawをQualityで再処理。
- 保存完了後に履歴へ追加。

### 3. Debug overlay強化

表示するもの:

- `profile`
- `processing_ms`
- `frame_age_ms`
- `landmark_age_ms`
- `mask_age_ms`
- `motion_factor`
- `published_frame_id`
- `analysis_frame_id`
- `discarded_processed_frames`

効果:

- 残像が出た時、どのageが悪いか即分かる。

## やらない方がよい、または後回しでよいもの

### PyPy

MediaPipe/OpenCV/NumPyのwheel互換性とC拡張依存が強く、現実的な効果が見込みにくいです。

判断:

- 採用しない。

### いきなりONNX/CoreML化

推論が支配的か分かる前に変換すると、変換・精度差・op対応に時間を使います。

判断:

- stage別計測で推論が主因と判明してから。

### 画質処理を全てGPUへ移す

OpenCV/NumPyとの往復転送が増えると逆に遅くなります。

判断:

- GPUへ移すなら、複数処理をまとめてGPU内で完結させる。

### 残像をblurでごまかす

残像は古いmask/古い加工結果の問題なので、blurを強めると顔がぼやけるだけです。

判断:

- 根本対策はage gatingとlatest-only publish。

## 推奨実装順

### Step 1: 計測と表示

1. `landmark_age_ms`, `mask_age_ms` を追加。
2. stage別timerを追加。
3. PerformanceパネルまたはDebug overlayにage/motionを表示。
4. benchmark scriptにstage別結果を出す。

受け入れ基準:

- 残像発生時に、古いrawなのか、古いlandmarkなのか、古いmaskなのか判断できる。

現状:

- `landmark_age_ms`, `mask_age_ms`, `publish_interval_ms`, `publish_lag_frames`, `preview_stall_ms` は `/api/performance` とUIのPerformanceパネルで確認できる。
- stage別timerとbenchmark scriptへの詳細traceは未実装。

### Step 2: preview/capture分離

1. 最新raw frameをruntimeが保持。
2. previewはFastを前提にする。
3. capture時だけQualityで再処理。
4. DB metadataにpreview/capture profileを残す。

受け入れ基準:

- preview設定を軽くしても、保存画像はQuality相当で残る。

現状:

- 最新raw frameの保持とcapture時Quality再処理は実装済み。
- DBには撮影時のQuality設定を保存済み。
- preview profileを別フィールドとして保存する部分は未実装。

### Step 3: motion時の厳格制御

1. motionが高い場合はpublish lag許容を0へ。
2. segmentation由来補正をmotion中だけ弱める。
3. motion中はlandmark-only maskへ切り替える。

受け入れ基準:

- 顔を左右に振った時、肌/背景の残像が前位置に残らない。

### Step 4: pipeline分離

1. Capture/Analyze/Render/Publishを分ける。
2. 各queueをサイズ1にする。
3. analysis resultにsource frame idを持たせる。
4. rendererでage gatingする。

受け入れ基準:

- 推論が詰まってもカメラreadと表示が詰まらない。

### Step 5: `LIVE_STREAM`移行

1. Face Landmarkerをasync callback化。
2. Segmenterをasync callback化。
3. result callbackは保存だけにし、重い変換はrenderer側へ渡す。

受け入れ基準:

- MediaPipe推論中にアプリ全体が待たない。

### Step 6: hot path削減

1. ROI化。
2. blend集約。
3. bilateral削減。
4. 動的process_width。
5. background/glow低頻度化。

受け入れ基準:

- Fast平均70ms以下。
- `frame_age_ms` 100ms以下。

### Step 7: native/GPU検証

1. それでも不足ならRust/PyO3またはC++/pybind11でblend/warpをnative化。
2. Mac前提が固まるならCore Image/Metalでblur/blend/glowをGPU化。
3. 推論が支配的ならONNX/CoreMLまたはLiteRT GPUを検証。

受け入れ基準:

- Fast preview 15fps以上、できれば20fps以上。
- 残像が出ないことをFPSより優先する。

## 具体的な設計メモ

### `PerformanceSummary` 拡張案

```python
class PerformanceSummary(BaseModel):
    processing_ms: float
    encode_ms: float
    effective_fps: float
    dropped_frames: int
    discarded_processed_frames: int
    frame_age_ms: float
    landmark_age_ms: float
    mask_age_ms: float
    motion_factor: float
    publish_interval_ms: float
    publish_lag_frames: int
    preview_stall_ms: float
    published_frame_id: int
    latest_raw_frame_id: int
    profile: Literal["quality", "fast"]
```

### analysis result案

```python
@dataclass(frozen=True)
class AnalysisPacket:
    source_frame_id: int
    source_captured_at: float
    analyzed_at: float
    detections: FaceDetections
    masks: SegmentationMasks | None
```

### age gating案

```python
def analysis_strength(age_ms: float, full_until: float, zero_at: float) -> float:
    if age_ms <= full_until:
        return 1.0
    if age_ms >= zero_at:
        return 0.0
    return 1.0 - (age_ms - full_until) / (zero_at - full_until)
```

使用例:

- 目拡大/小顔: `landmark_age_ms` で制御。
- 肌/髪/背景: `mask_age_ms` で制御。
- glow: `frame_age_ms` と `motion_factor` で制御。

### publish判定案

```python
def should_publish(packet, latest_raw_id, frame_age_ms, motion_factor):
    max_lag = 1 if motion_factor < 0.04 else 0
    if latest_raw_id - packet.id > max_lag:
        return False
    if frame_age_ms > 120:
        return False
    return True
```

## 検証方法

### 自動ベンチ

```bash
uv run python scripts/benchmark_pipeline.py --profile both --iterations 30 --warmup 5
```

追加したいオプション:

```bash
uv run python scripts/benchmark_pipeline.py --profile fast --trace-stages --simulate-motion
```

見る値:

- avg / p95 / max
- stage別ms
- frame age
- landmark age
- mask age
- discarded processed frames

### 手動検証

1. `uv run uvicorn purikura_test.app:app --reload`
2. UIで `Processing profile: Fast` を選ぶ。
3. Debug overlayを `All` にする。
4. 顔を左右に速く動かす。
5. 以下を確認する。
   - 前の顔位置に白肌や背景白寄せが残らない。
   - 動いている間は補正が薄くなってもよい。
   - 止まったら高品質な補正に戻る。
   - `frame_age_ms`, `landmark_age_ms`, `mask_age_ms` が目標範囲に戻る。

### 合格基準

- 速い顔移動時に、顔の前位置へ肌mask/背景maskの残像が残らない。
- Fast previewで平均15fps以上。
- Fast previewで `frame_age_ms <= 100ms` が概ね維持される。
- 静止時にはプリクラ風の見た目が戻る。
- 撮影保存はQuality処理でDBに残る。

## 結論

次にやるべき順番は、native化やONNX化ではなく、以下です。

1. `landmark_age_ms` / `mask_age_ms` を計測してUIに出す。
2. previewとcaptureを分離し、previewは低遅延、captureはQualityにする。
3. motionが大きい時は古い加工結果をpublishせず、segmentation由来補正をさらに弱める。
4. Capture/Analyze/Render/Publishを分け、各段をlatest-onlyにする。
5. MediaPipeを `LIVE_STREAM` modeへ移す。
6. その後にROI化、blend集約、native化、Core Image/Metal、ONNX/CoreMLを計測結果に基づいて選ぶ。

残像を消す上で最も重要なのは、「重い処理を速くする」ことだけではなく、「古い情報を表示に使わない」ことです。
