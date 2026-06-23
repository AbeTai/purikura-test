# リアルタイム追従性・高速化ロードマップ

## 背景

現状の課題は「単にFPSが低い」だけではなく、顔が動いたときに古い解析結果や古い加工結果が残り、加工が残像のように見えることです。プリクラ補正では顔ランドマーク、人物セグメンテーション、局所変形、肌/髪/背景マスク、MJPEG配信が直列に近い形で関わるため、以下を分けて考えます。

- **処理時間**: 1フレームの加工に何msかかるか。
- **解析鮮度**: 今表示している映像に対して、顔ランドマーク/マスクが何フレーム前のものか。
- **表示遅延**: カメラ取得からブラウザ表示まで何ms遅れているか。
- **空間追従**: 顔が動いたとき、古いマスクを現在フレームへどれだけ正しく合わせられているか。

このドキュメントでは、今後取り得る高速化と残像対策を網羅的に整理します。フォールバック実装は不要という方針に合わせ、必須モデルや必須処理が失敗した場合は明示的に止める前提です。

## 公式ドキュメント上の根拠

- MediaPipe Face Landmarker は `VIDEO` / `LIVE_STREAM` モードでトラッキングを使い、毎フレームのモデル実行を避けて遅延を下げられる。`LIVE_STREAM` は非同期で、処理中に新しい入力が来るとその入力を無視するため、低遅延設計に使える。  
  <https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python>
- MediaPipe Image Segmenter も `IMAGE` / `VIDEO` / `LIVE_STREAM` の実行モードを持ち、`segment`, `segment_for_video`, `segment_async` を使い分けられる。  
  <https://developers.google.com/mediapipe/solutions/vision/image_segmenter/python>
- MediaPipe Framework の real-time streams は、リアルタイム処理ではtimestamp boundsを使って下流処理を待たせない設計が重要だと説明している。  
  <https://developers.google.com/mediapipe/framework/framework_concepts/realtime_streams>
- OpenCVは、まず計測し、OpenCV最適化を有効化し、NumPyのPythonループよりOpenCV/ベクトル化処理を使う方針を推奨している。  
  <https://docs.opencv.org/4.x/dc/d71/tutorial_py_optimization.html>
- LiteRT GPU delegate はGPUが並列ML推論で低遅延化に有効だが、非対応opが混じるとCPU/GPU同期コストで逆に遅くなることがある。  
  <https://developers.google.com/edge/litert/performance/gpu>
- ONNX Runtime CoreML Execution Provider はmacOS/iOSでCore ML経由の実行を可能にするが、モデル変換・対応op・精度差・初回コンパイルを評価する必要がある。  
  <https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html>
- Apple Core Image の性能指針では、コンテキスト再利用、小さい画像での処理、CPU/GPU間転送の回避、不要な色管理の回避が重要とされている。  
  <https://developer.apple.com/library/archive/documentation/GraphicsImaging/Conceptual/CoreImaging/ci_performance/ci_performance.html>
- MetalはApple silicon上で低オーバーヘッドなGPU制御、compute、機械学習統合、プロファイリングを提供する。  
  <https://developer.apple.com/metal/>

## 現状コードから見えるボトルネック

対象コード:

- `src/purikura_test/runtime.py`
- `src/purikura_test/effects.py`
- `scripts/benchmark_pipeline.py`

現状の主要な遅延要因:

1. **加工結果が古くなる可能性**
   - `PurikuraRuntime` はカメラreadと加工を別スレッドに分け、`_pending_frame` を1枚だけ保持している。
   - 加工中に新しいフレームが来た場合、古いpendingは捨てるが、すでに加工中のフレームは最後まで処理される。
   - 重いフレームの処理が完了すると、その時点ではすでに古い顔位置の加工済み画像が最新として表示される。

2. **Fast版でも顔/マスクの解析が間引かれている**
   - `CachedFaceTracker(... detect_every_n_frames=2)` により顔ランドマークは最大1フレーム古い。
   - `HeadSegmenter(segment_every_n_frames=8, ema_alpha=0.72)` によりセグメンテーションは最大7フレーム古い。
   - 顔が速く動くと、マスクだけ前の位置に残りやすい。

3. **検出結果キャッシュが残像を助長する**
   - `MediaPipeFaceTracker(max_cached_seconds=0.25)` は検出0件でも直近顔を短時間返す。
   - 顔が外れた/大きく移動した瞬間に、旧bbox/landmarksが補正に使われる可能性がある。

4. **EMA maskが位置ずれを平均してしまう**
   - `ema_masks(previous, current, alpha)` は前回maskと今回maskを画素位置のまま混ぜる。
   - 顔が動いた場合、位置合わせせずに混ぜるため、前位置の肌/髪/背景補正が薄く残る。

5. **画像処理が多段でarray copyも多い**
   - bilateral/blur、Lab/HSV変換、複数の`blend_by_mask`、局所変形、リサイズ、JPEG encodeが続く。
   - `result.copy()` やfull-frame overlayが多く、CPUメモリ帯域を使う。

6. **MJPEGは単純だが、ブラウザ表示としては最適ではない**
   - JPEG encode自体は以前の計測では支配的ではないが、MJPEGはフレーム同期、バックプレッシャ、表示遅延の制御が弱い。

## まず測るべき指標

高速化前に、残像の原因を分解して測る必要があります。平均msだけでは判断できません。

追加したいメトリクス:

- `capture_timestamp_ms`: カメラread時刻。
- `analysis_timestamp_ms`: landmarker/segmenterの解析対象フレーム時刻。
- `processed_timestamp_ms`: 加工完了時刻。
- `published_timestamp_ms`: `_latest_jpeg` 更新時刻。
- `frame_age_ms`: 表示時点での元フレーム年齢。
- `landmark_age_ms`: 表示時点での顔ランドマーク年齢。
- `mask_age_ms`: 表示時点でのセグメンテーションmask年齢。
- `motion_px`: 前回顔中心から今回顔中心までの移動量。
- `discarded_inflight_frames`: 加工完了したが古すぎて捨てたフレーム数。
- stage別時間: camera read, resize, landmarker, segmenter, warp, skin, makeup, background, glow, frame composite, jpeg encode。

受け入れ目安:

- Fast preview: `frame_age_ms <= 100ms` を優先。FPSより重要。
- `landmark_age_ms <= 66ms`、`mask_age_ms <= 150ms` を目標。
- 顔中心移動が大きいときは、古いmaskを使って画質を保つより、補正を弱めて残像を消す方を優先。

## 短期: 残像に直接効く改善

### 1. 古い加工結果を公開しない

加工開始時にframe sequence idを持たせ、完了時点で最新read frameとの差が大きければ `_latest_processed` を更新しない。

案:

- `FramePacket(id, captured_at, frame)` を導入。
- `_pending_frame` ではなく `_latest_raw_packet` を保持。
- processorは取得したpacketを処理する。
- 処理完了時に `latest_raw_id - packet.id > threshold` なら破棄。
- thresholdはFastなら0または1、Qualityなら1から開始。

効果:

- 古い顔位置の加工結果が表示される問題を直接減らせる。
- FPSは下がる可能性があるが、体感遅延は改善する。

優先度: 最優先。

### 2. 検出キャッシュのTTLを短縮し、動きが大きいとき無効化する

現状の `max_cached_seconds=0.25` は顔が速く動くと長いです。Fast previewでは50-100ms程度から始めるのが妥当です。

案:

- Fast版は `max_cached_seconds=0.08`。
- 前回bbox中心と現在の粗い顔bbox推定との差が大きい場合はキャッシュを使わない。
- 顔検出0件時に旧landmarksを使う場合、補正強度を時間でフェードアウトする。

効果:

- 顔が外れた/高速移動した瞬間の「前の顔補正」が残りにくい。

優先度: 最優先。

### 3. EMA maskを「位置合わせなし混合」から「動き対応」に変える

現状のEMAは画素位置固定なので、動きに弱いです。

段階案:

1. 顔移動が一定以上ならmask EMAをリセットする。
2. bbox中心移動分だけ前回maskを平行移動してからEMAする。
3. 可能なら目/鼻/口などのランドマーク対応点からaffine transformを推定して前回maskをwarpする。

効果:

- セグメンテーションを8フレームごとにしても、maskが顔に追従しやすくなる。

優先度: 最優先。

### 4. 動きが大きい間だけ補正を軽くする

「完全な高品質補正が遅れて付く」より、「動いている間は補正を薄くして残像なし」の方がプレビュー体験は良いです。

案:

- `motion_px / face_width` から `motion_factor` を作る。
- `motion_factor` が高いとき:
  - skin smoothingを下げる。
  - background_high_keyを下げる。
  - soft_glowを下げる。
  - segmentation由来maskではなくlandmark由来の小さめface maskだけ使う。

効果:

- 顔が止まった瞬間に高品質補正へ戻せる。
- 撮影時は静止を促すUIと相性が良い。

優先度: 高。

### 5. セグメンテーション更新頻度を固定ではなく適応制御にする

今はFastで8フレームごとです。顔が静止している時は8-12でもよいが、動いた瞬間は毎フレームまたは2フレームごとに戻すべきです。

案:

- `motion_factor < low`: segment every 10-12。
- `low <= motion_factor < high`: segment every 4。
- `motion_factor >= high`: segment every 1-2、ただし処理詰まり時は古い結果を使わず補正を弱める。

効果:

- 静止時の軽さと動作時の追従性を両立できる。

優先度: 高。

## 短期: Python/OpenCV内での高速化

### 6. Fast版の処理解像度を動的に下げる

現在は横幅640px固定です。残像が出るほど処理が詰まる場合は、顔が動いている間だけ480pxや512pxへ落とす選択肢があります。

案:

- `process_width=640` を通常値。
- 直近 `processing_ms > 80ms` または `frame_age_ms > 120ms` なら `process_width=512`。
- 顔が安定して `processing_ms < 50ms` なら640へ戻す。

副作用:

- 目メイクや輪郭補正の精度は下がる。
- 撮影保存はQualityに切り替える、またはcapture時のみ高解像度再処理する設計が必要。

優先度: 高。

### 7. full-frame blendをROI化する

Fast版でも `apply_hair_silk`, `apply_background_high_key`, `apply_soft_glow` は全体処理を含みます。

案:

- 顔/髪/肌は `head_roi` のみ。
- 背景白寄せは低解像度maskで計算し、最終blendを1回にまとめる。
- soft_glowは毎フレームではなく2-3フレームごとに低解像度更新し、間は前回glowを使う。ただし動きが大きいときはglowを弱める。

優先度: 高。

### 8. blend回数を1回へ集約する

現在は複数の `blend_by_mask` が段階的に走り、そのたびにarrayを作ります。

案:

- 各処理で「target layer」と「alpha mask」を作る。
- 最終的に `composite_accumulator` で1-2回の合成にまとめる。
- `np.float32` 変換を各関数で繰り返さず、必要区間だけ共通化する。

優先度: 中。

### 9. bilateral filterを削減/置換する

bilateralは肌には効きますが重いです。

案:

- Fast版はbilateralではなく `GaussianBlur + edge-preserving mask` へ寄せる。
- 顔ROIのskin mask内だけに限定する。
- 連続フレームではblur結果を再利用し、動きが少ない時だけ更新する。
- Qualityだけbilateralを残す。

優先度: 中。

### 10. OpenCV最適化設定とスレッド数を明示する

確認項目:

- `cv2.useOptimized()` がTrueか。
- `cv2.setUseOptimized(True)` を起動時に呼ぶ。
- `cv2.getNumThreads()` / `cv2.setNumThreads(n)` を計測し、MediaPipeと取り合わない値を探る。

注意:

- スレッド数を増やせば必ず速いわけではない。MediaPipeやUvicornとの競合で遅くなる場合がある。

優先度: 中。

### 11. 不要な色空間変換を削る

現状はBGR/RGB、Lab、HSV変換が複数回出ます。

案:

- 1フレーム内でLab/HSV変換を共有する。
- tone調整はLUT化する。
- RGB変換はMediaPipe入力用だけに限定し、OpenCV処理はBGRのまま閉じる。

優先度: 中。

## 中期: 非同期パイプライン化

### 12. MediaPipeをLIVE_STREAM modeへ移行する

Face LandmarkerとImage Segmenterを `LIVE_STREAM` にし、`detect_async` / `segment_async` のcallbackで最新結果だけを保持します。

設計案:

- camera thread: frame packetを生成。
- analyzer thread/callback:
  - Face Landmarker LIVE_STREAMへ最新packetを投入。
  - Segmenter LIVE_STREAMへ必要時だけ投入。
  - callbackは `analysis_result_by_frame_id` に保存。
- renderer/processor thread:
  - 最新raw frameを取得。
  - 最新のlandmarks/masksをtimestamp差で評価。
  - 古すぎる解析結果は使わず、補正を弱める。

効果:

- MediaPipe推論でprocessorがブロックされにくくなる。
- 処理中の新入力をMediaPipe側が無視できるため、キュー肥大を避けやすい。

注意:

- callback結果とraw frameの同期設計が必要。
- `LIVE_STREAM` は結果が必ず全フレーム返るわけではない。返らないフレームを前提にUI/補正を設計する。

優先度: 高。

### 13. カメラ、解析、描画、配信を4段に分ける

現状はread threadとprocessor threadです。残像対策として、解析と描画を分けます。

推奨構成:

- Capture: 最新raw frameだけを持つ。
- Analyze: 最新rawからlandmarks/masksだけを更新。古い入力は捨てる。
- Render: 最新rawと最新analysisを使い、age gatingして加工。
- Publish: JPEG/WebSocket/WebRTC用に配信。古いrenderは捨てる。

各段のキューは原則サイズ1。処理が詰まったら古いものを捨て、順序保証より最新性を優先します。

優先度: 高。

### 14. timestampベースのage gatingを徹底する

landmarks/masksに `source_frame_id` と `source_timestamp` を持たせます。

ルール例:

- `landmark_age_ms > 100`: 目拡大/小顔/アイメイクを停止。
- `mask_age_ms > 180`: segmentation由来の肌/背景/髪補正を停止または弱める。
- `frame_age_ms > 120`: 加工結果をpublishしない。

効果:

- 残像を「見た目の問題」ではなく「古い情報を使わない制約」として制御できる。

優先度: 高。

## 中期: 追従性を上げるアルゴリズム

### 15. ランドマークからmaskをワープする

SelfieMulticlassのmaskは更新を間引きたいが、顔は動く。そこで前回maskを現在ランドマークへwarpします。

案:

- 顔の代表点: 両目中心、鼻先、口中心、顎、左右頬。
- 前回代表点と今回代表点からaffine/piecewise affineを推定。
- 前回 `skin/hair/head/protected` maskをwarp。
- 新segmentationが来たら補正。

優先度: 高。

### 16. Optical flowでmaskを移動する

ランドマークが不安定なときは、Lucas-Kanade optical flowやdense optical flowでmaskを追従できます。

案:

- 顔ROI内の特徴点を追跡。
- median flowでmaskを平行移動/affine変換。
- flow信頼度が低い場合はmaskを捨てる。

副作用:

- 低照度やモーションブラーに弱い。
- 実装・検証コストはランドマークwarpより高い。

優先度: 中。

### 17. 顔が動く間はlandmark-only skin maskへ切り替える

高速移動時にセグメンテーションを追わせるのではなく、顔楕円と近傍body-skin推定だけで簡易補正にします。

案:

- `motion_factor >= high` ならSelfieMulticlass maskを使わない。
- 顔楕円を少し膨張し、目/眉/唇をprotect。
- 顔が静止したらsegmentation maskへ戻す。

効果:

- 動き中の残像をかなり減らせる。
- 耳/髪際の精度は下がるが、動いている間は気づきにくい。

優先度: 高。

## 中期: 出力方式の改善

### 18. MJPEGからWebRTCまたはWebSocket + WebCodecsへ移行する

MJPEGは実装が簡単ですが、リアルタイム映像としてはフレーム制御が弱いです。

候補:

- WebRTC:
  - 低遅延映像配信に向く。
  - ブラウザ側のバッファ制御が映像用途に最適化されている。
  - 実装は増える。
- WebSocket + JPEG/WebP:
  - 現状から移行しやすい。
  - クライアント側で古いframe idを捨てられる。
- WebCodecs:
  - ブラウザ対応と実装コストの確認が必要。

優先度: 中。

### 19. ブラウザ側で「最新フレームのみ描画」にする

MJPEGのままだとブラウザ内部バッファが見えません。WebSocket化すれば、JS側で古いframe idを破棄できます。

優先度: 中。

## 中期: 撮影時だけ高品質にする設計

ライブプレビューと撮影画像を同じ品質で処理しようとすると、プレビュー追従性が犠牲になります。

案:

- Preview:
  - process_width 480-640。
  - segmentationは適応更新。
  - 動き中は補正薄め。
- Capture:
  - 撮影ボタン押下時に最新rawを1枚保持。
  - その静止画をQualityで再処理してDB保存。
  - UIには「保存処理中」を出す。

効果:

- プレビューの残像対策と保存画質を分離できる。

優先度: 高。

## 長期: ネイティブ化

### 20. Rust/PyO3またはC++/pybind11でhot pathを切り出す

Python関数呼び出しとNumPy array生成が多い箇所をnative化します。

候補:

- `blend_by_mask`
- 複数maskの合成
- ROI crop/composite
- tone LUT
- EMA/warp mask
- local warp/zoom
- debug overlay以外の描画レイヤ合成

Rust/PyO3:

- 安全性とmaturin/uv管理の相性が良い。
- ndarray/numpy連携設計が必要。

C++/pybind11:

- OpenCV C++との相性が良い。
- 既存OpenCV処理をC++へ寄せやすい。

優先度: 中。

### 21. OpenCV C++またはG-API化

PythonからOpenCV C++関数を呼ぶだけでも速い部分はありますが、複数処理をPythonで接着しているため中間コピーが増えます。C++側でROI処理をまとめると効果が出ます。

優先度: 中。

### 22. Apple Core Image / Metalへ移す

Mac前提なら、画像フィルタ・blend・blur・toneをGPUへ寄せる選択肢があります。

候補:

- Core Image:
  - blur, color matrix, blend, LUT系に向く。
  - `CIContext` 再利用、低解像度処理、CPU/GPU転送削減が重要。
- Metal:
  - mask blend、warp、tone、glowを1-2個のcompute shaderにまとめられる。
  - 実装コストは高いが、Apple siliconでは長期的に最も伸びしろがある。

優先度: 中から低。Pythonプロトタイプの後。

## 長期: 推論基盤の変更

### 23. MediaPipe TasksからMediaPipe Framework graphへ移行

Python Tasks APIではなく、MediaPipe graphとしてcamera, landmarker, segmenter, renderer相当を組むと、timestamp boundsやcalculator schedulingを明示できる。

効果:

- リアルタイムストリームとして自然な設計になる。
- 不要なPython境界を減らせる可能性がある。

コスト:

- C++/Bazel/graph知識が必要。
- 現アプリ構成からの移行が大きい。

優先度: 低から中。

### 24. Segmenterモデルを軽量化/置換する

SelfieMulticlassは便利ですが、残像対策では「正確なmask」より「最新のmask」が重要な場面があります。

候補:

- より軽量なperson/face parsingモデル。
- 顔近傍だけを対象にした小型segmenter。
- hair/backgroundを諦め、landmark + skin color modelで近似。
- 量子化TFLiteモデル。

注意:

- フォールバックではなく、選択したモデルを必須にする。
- 精度、速度、ライセンス、Mac wheel/Runtime対応を評価する。

優先度: 中。

### 25. ONNX/CoreML化

MediaPipe推論が支配的になった場合のみ検討します。現在の主問題は古い解析結果と画像処理遅延なので、最初にやるべきではありません。

検証項目:

- Face Landmarker相当をONNX/CoreMLにできるか。
- SelfieMulticlass相当の変換可否。
- CoreML Execution Providerの対応op。
- 初回compile時間。
- CPU/GPU/ANEの実効latency。
- 変換後のランドマーク座標やmask品質の差分。

優先度: 低から中。

## UI/UX側の対策

### 26. 撮影時に静止を促す

技術的な高速化とは別に、撮影体験としては「止まった瞬間に高品質」を狙う方が安定します。

案:

- 顔のmotionが低いときに「Ready」を出す。
- motionが高いときは「動いています」と表示し、撮影ボタンを一時的に弱める。
- カウントダウン中はQualityではなくFastで追従し、シャッター時にQuality再処理。

優先度: 中。

### 27. Debug overlayにageとmotionを出す

残像を調整するには、画面上で何が古いか分かる必要があります。

表示案:

- `frame_age=xxms`
- `landmark_age=xxms`
- `mask_age=xxms`
- `motion=0.xx`
- `published_frame_id=...`
- `analysis_frame_id=...`

優先度: 高。

## 実装済み

2026-06-23時点で、Phase 1の一部として以下を実装済みです。

- カメラread時に `FramePacket(id, captured_at, frame)` を付与。
- 加工完了時点で `latest_raw_frame_id - packet.id` が閾値を超えていた場合、古い加工結果をpublishせず破棄。
- Fast previewは1フレーム超過、Qualityは2フレーム超過を破棄対象にする。
- `/api/performance` に `discarded_processed_frames`, `frame_age_ms`, `published_frame_id`, `latest_raw_frame_id` を追加。
- Face Landmarkerの検出0件時キャッシュTTLを250msから100msへ短縮。
- 顔中心移動が顔幅の10%を超えた場合、segmentation maskの再利用とEMAを止め、新しいmaskを使う。
- 顔中心移動が10%未満の場合、前回maskを現在の顔中心へ平行移動してから再利用/EMAする。
- Fast previewで顔ランドマークを毎フレーム更新し、`motion_factor` を `/api/performance` とDebug overlayへ表示。
- `motion_factor` が高い間は、目/小顔変形、肌、メイク、背景、グローの強度を一時的に弱めて、古い位置の加工が強く残らないようにする。

未実装で次に優先すべきものは、`landmark_age_ms`, `mask_age_ms` の計測、maskのaffine warp、撮影時だけQuality再処理する分離設計です。

## 優先実装順

### Phase 1: 残像の直接除去

1. `FramePacket` と sequence id を入れる。
2. 加工完了時に古いpacketをpublishしない。
3. `landmark_age_ms`, `mask_age_ms`, `frame_age_ms`, `motion_factor` を計測する。
4. 顔移動が大きいときはmask EMAをリセットする。
5. Fast版の `max_cached_seconds` を短縮する。
6. Debug overlayにage/motionを表示する。

期待効果:

- 古い加工結果の表示が減る。
- 残像の原因をメトリクスで追える。
- 見た目が一瞬薄くなる場面はあるが、残像より自然。

### Phase 2: 動きに追従するmask

1. 顔中心移動で前回maskを平行移動してからEMA。
2. ランドマーク代表点でaffine warp。
3. 動き量に応じたsegmentation更新頻度。
4. 動き中はlandmark-only maskへ切り替え。

期待効果:

- 顔移動時の肌/背景/髪補正のズレが減る。

### Phase 3: Fast版の処理削減

1. 動的process_width。
2. full-frame処理のROI化。
3. soft_glow/backgroundの低解像度・低頻度化。
4. blend集約。
5. bilateral削減。

期待効果:

- previewを15fps以上に戻しやすい。

### Phase 4: 非同期推論

1. Face LandmarkerをLIVE_STREAMへ移行。
2. Image SegmenterをLIVE_STREAMへ移行。
3. analysis callbackとrendererをtimestamp同期。
4. 古い解析結果のage gatingを徹底。

期待効果:

- 推論待ちによるフレーム停滞を減らせる。

### Phase 5: ネイティブ/GPU化

1. Rust/PyO3またはC++/pybind11でmask blend/warpをnative化。
2. Core Imageでblur/tone/blendをGPU化。
3. Metal computeでwarp/blend/glowを統合。
4. 推論が支配的になったらONNX/CoreML/LiteRT GPUを検証。

期待効果:

- 高品質補正を保ったままプレビューを軽くできる。

## 実装時の注意

- フォールバックは追加しない。必須モデルや必須runtimeがない場合は明示エラーにする。
- Fast previewは「最新性」を最優先し、古い結果を表示しない。
- Quality captureは「画質」を優先し、必要なら撮影後に別処理として待たせる。
- FPSだけで判断しない。`frame_age_ms` と `mask_age_ms` を必ず見る。
- 最適化ごとに `scripts/benchmark_pipeline.py` だけでなく、カメラ実機での `/api/performance` とDebug overlayを確認する。
- 残像対策では、古いmaskを使って補正品質を維持するより、補正を一時的に弱める判断を優先する。
