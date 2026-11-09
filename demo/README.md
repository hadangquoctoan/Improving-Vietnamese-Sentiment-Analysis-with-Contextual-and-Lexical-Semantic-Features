# Demo API — Sentiment Lens (Report 4)

Backend HTTP cho demo: nhận câu tiếng Việt thô, chạy checkpoint đã train của nhóm, trả nhãn, xác suất và số liệu theo từng từ (gate, lexicon 12 chiều). Không sửa code lõi trong `src/`.

## Cài đặt trên máy mới (Windows)

1. **Python 3.12** và một môi trường riêng (conda hoặc venv). Cài torch theo GPU trước, rồi các gói còn lại:
   ```powershell
   pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU NVIDIA (RTX 50xx cần cu128+); không có GPU: .../whl/cpu
   pip install -r demo/requirements-demo.txt
   ```
2. **JDK 17 trở lên** (VnCoreNLP chạy bằng Java), ví dụ Temurin 21. Đặt biến môi trường `JAVA_HOME` trỏ tới thư mục JDK.
3. **Tài nguyên từ Drive của nhóm** (không nằm trong git), đặt đúng đường dẫn trong repo:

   | Đường dẫn trong repo | Lấy từ Drive |
   |---|---|
   | `artifacts/checkpoints/*.pt` (5 file, mỗi file ~545 MB) | artifacts → `checkpoints/` |
   | `artifacts/hf-cache/phobert-base/` | artifacts → `hf-cache/phobert-base/` |
   | `artifacts/tools/vncorenlp/` | artifacts → `tools/vncorenlp/` |
   | `data/lexicons/composite_12d/v0.2.0/` | data → `lexicons/composite_12d/v0.2.0/` |

   File checkpoint lớn nên tải bằng trình duyệt đã đăng nhập Google; `gdown` thường bị Drive chặn vì giới hạn lượt tải. Tối thiểu cần `best_fused.pt`; thiếu checkpoint nào thì model đó hiện "Thiếu checkpoint" trong menu.
4. Kiểm tra (tùy chọn, cần `JAVA_HOME`): `python -m pytest demo/tests -q` → 11 passed.

## Chạy

**Cách dễ nhất (nhấp đúp, trong thư mục `demo/` của repo):**
- `START_DEMO.bat`: chỉ máy này, mở `http://127.0.0.1:8000`.
- `START_DEMO_LAN.bat`: thiết bị khác cùng Wi-Fi/LAN mở được. Cửa sổ in địa chỉ `http://<IP-máy-này>:8000`.
- Giữ cửa sổ đen mở suốt buổi demo; đóng cửa sổ hoặc Ctrl+C là tắt server. Chờ ~10–20 s cho model nạp xong rồi mới mở trình duyệt.
- Script tự tìm Java (`DEMO_JAVA_HOME` → `JAVA_HOME` → `D:\tools\jdk21`), Python (`DEMO_PYTHON` → `D:\envs\nlpdevelop` → `python` trong PATH) và báo rõ nếu thiếu file tài nguyên nào.
- Nếu báo "Port 8000 đang bị chiếm": demo đã chạy ở cửa sổ khác; đóng cửa sổ đó (hoặc chạy `run_api.ps1 -Port 8001`).
- Nếu repo nằm ở đường dẫn có dấu cách (ví dụ `C:\Users\Ten May\...`), VnCoreNLP báo lỗi "cache path contains whitespace": đặt `VNCORENLP_CACHE_DIR` tới một thư mục không dấu cách (ví dụ `D:\vncorenlp-cache`). Khi chạy bộ test của repo, đặt thêm `TMP`/`TEMP` tới thư mục không dấu cách.

**Mạng LAN:**
- Máy demo đã có luật Windows Firewall cho phép `Python312\python.exe` nhận kết nối (kiểm tra 2026-11-09), nên thường không cần làm gì thêm.
- Nếu máy khác vẫn không vào được, mở PowerShell **Run as administrator** và chạy:
  `New-NetFirewallRule -DisplayName "Sentiment Lens demo 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private,Public`
  (xóa sau buổi demo: `Remove-NetFirewallRule -DisplayName "Sentiment Lens demo 8000"`).
- Wi-Fi trường thường chặn các máy nói chuyện với nhau (client isolation). Khi đó dùng hotspot điện thoại cho cả máy demo và thiết bị xem.
- Không có đăng nhập: ai trong cùng mạng cũng dùng được. Chỉ bật chế độ LAN trong mạng tin cậy, tắt sau buổi demo. Server chỉ phát trang demo, API và thư mục `assets/` cạnh trang.

**Giao diện:** `demo/web/index.html`, được server phục vụ ở `/` (đổi bằng biến `DEMO_WEB_FILE`). Mở file HTML trực tiếp (`file://`) sẽ không gọi được API.

**Chạy bằng lệnh** (thay cho file `.bat`):

```powershell
powershell -ExecutionPolicy Bypass -File demo\run_api.ps1          # chỉ máy này
powershell -ExecutionPolicy Bypass -File demo\run_api.ps1 -Lan     # cho mạng LAN
```

- API: `http://127.0.0.1:8000`, tài liệu tương tác: `http://127.0.0.1:8000/docs`.
- Chỉ chạy 1 worker: VnCoreNLP chạy trong JVM, mỗi tiến trình chỉ khởi động JVM được một lần; request được xếp hàng qua một khóa.
- CORS mở cho `http://localhost:*` và `http://127.0.0.1:*` (dev server của frontend); khi mở qua server thì trang và API cùng nguồn, không cần CORS.

## API

| Method | Path | Trả về |
|---|---|---|
| GET | `/api/health` | `{"status":"ok","device":"cuda","loaded":["uit"]}` |
| GET | `/api/models` | `[{"id":"uit","name":"UIT-VSFC","description":"…","labels":["neg","neu","pos"],"recommended":true,"available":true,"loaded":true}, …]` |
| GET | `/api/samples` | Danh sách câu mẫu đã chạy thật |
| POST | `/api/predict` | Kết quả phân tích (dưới) |

Request:

```json
{ "text": "Thầy cô hướng dẫn rất tận tâm", "model": "uit", "lexicon": true }
```

`model`: một `id` trong bảng dưới (mặc định `"uit"`). `lexicon`: mặc định `true`.

### Mô hình

Cả 5 checkpoint đã kiểm trên máy demo: dung lượng khớp Drive, SHA-256 khớp nhật ký nhóm (`plans-and-report`), dev tính lại khớp số lưu trong checkpoint.

| `id` | Checkpoint | Nhãn | `recommended` | Số liệu nhóm ghi |
|---|---|---|---|---|
| `uit` | `best_fused.pt` | 3 | có | dev macro-F1 0,8685; test một lần 0,8451 |
| `vlsp` | `vlsp2016_scratch_best.pt` | 3 | có | dev 0,7280; test 0,7532 (exploratory) |
| `aivivn` | `aivivn2019_best.pt` | **2** (`neg`, `pos`) | có | dev 0,9074; test 0,9131 (split nội bộ) |
| `uit-alt` | `best.pt` | 3 | không | dev 0,8814; **chưa chạy test UIT** |
| `vlsp-run1` | `vlsp2016_best.pt` | 3 | không | lần train đầu trên VLSP2016 từ PhoBERT gốc (không phải từ `best.pt`); test 0,7532 |

`vlsp` và `vlsp-run1` cùng công thức, cùng seed; bản scratch là lần chạy lại và cho dự đoán trùng tới 6 chữ số thập phân (khác byte do CUDA). Coi như một model.

Frontend hiện cả 5 model, chia nhóm "Khuyên dùng" (`recommended: true`, mỗi miền một model) và "Đối chiếu". Server giữ tối đa 3 model trong VRAM; chọn model thứ 4 sẽ gỡ model dùng lâu nhất. Nạp một model lần đầu (hoặc sau khi bị gỡ) mất 2–3 s, các câu sau ~10–30 ms.

Response thật: xem `example_response.json` (sinh từ server ngày 2026-11-09). Rút gọn:

```json
{
  "label": "Tích cực", "label_key": "pos", "label_id": 2,
  "probs": { "neg": 0.001159, "neu": 0.000364, "pos": 0.998478 },
  "text": "Thầy cô hướng dẫn rất tận tâm",
  "segmented": "thầy_cô hướng_dẫn rất tận_tâm",
  "truncated": false,
  "lexicon": true,
  "tokens": [
    { "word": "thầy_cô", "in_lexicon": false, "gate": null, "lexicon_ratio": null, "dims": null, "subwords": 1 },
    { "word": "tận_tâm", "in_lexicon": true, "gate": 0.3165, "lexicon_ratio": 0.507,
      "dims": { "polarity": null, "valence": null, "arousal": null, "dominance": null,
                "joy": 0.0, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "disgust": 0.0,
                "surprise": 0.0, "trust": 1.0, "anticipation": 0.0 }, "subwords": 1 }
  ],
  "coverage": { "words": 4, "in_lexicon": 2 },
  "model": { "id": "uit", "name": "UIT-VSFC", "checkpoint": "best_fused.pt", "dataset": "UIT-VSFC",
             "lexicon_version": "v0.2.0", "epoch": 4, "dev_macro_f1": 0.8685 },
  "latency_ms": 95
}
```

Lỗi luôn có dạng `{"error": "<câu báo tiếng Việt cho người dùng>"}`:
- `400`: câu rỗng, dài quá 1.000 ký tự, mô hình không tồn tại.
- `422`: body không phải JSON đúng định dạng.
- `503`: thiếu checkpoint trên máy.

## Ý nghĩa các trường

- `probs`: khóa theo `labels` của model. Model `aivivn` chỉ có `neg` và `pos`, **không có `neu`**: giao diện phải ẩn "Trung tính" và vẽ thanh 2 đầu khi chọn model này.
- `segmented`: câu sau chuẩn hóa (NFC, chữ thường) và tách từ VnCoreNLP; `_` nối từ ghép. Hiển thị `word` thì thay `_` bằng khoảng trắng.
- `in_lexicon`: từ có trong lexicon 12 chiều `v0.2.0`. Từ ngoài lexicon: mô hình giữ nguyên biểu diễn PhoBERT, `gate`/`dims` là `null`.
- `dims`: 12 chiều lexicon (`polarity, valence, arousal, dominance, joy, sadness, anger, fear, disgust, surprise, trust, anticipation`). `polarity` theo thang gốc **[-1, 1]**; các chiều khác [0, 1]; 8 cảm xúc là 0/1. Chiều nguồn không có dữ liệu là `null` (khác với 0).
- `gate`: trung bình 768 chiều của cổng `g = σ(W[H; L̃] + b)` trên các subword của từ. Cách tóm tắt này do người làm demo chọn, cần nhóm duyệt trước khi đưa lên slide.
- `lexicon_ratio`: `‖g ⊙ L̃‖ / ‖H‖` trung bình trên subword: độ lớn phần lexicon được cộng vào so với biểu diễn PhoBERT của chính từ đó.
- `truncated`: `true` khi câu vượt 256 subword, phần đuôi bị bỏ (theo ranh giới từ); các từ bị bỏ có `subwords: 0`.
- `lexicon: false`: chạy chính checkpoint fused nhưng **tắt nhánh lexicon lúc suy luận**. Classifier chưa từng thấy đầu vào kiểu này, nên đây không phải baseline A (PhoBERT train riêng). Giao diện phải ghi rõ nếu có công tắc này.

## Lưu ý cho giao diện (số đo thật trên checkpoint UIT)

- **Gate gần như không đổi giữa các từ**: qua 34 từ trong lexicon ở 6 câu mẫu, `gate` chỉ từ 0,27 đến 0,34 (khởi tạo là sigmoid(−1) ≈ 0,27). Tô đậm nhạt theo `gate` sẽ gần như đều màu. Dùng `lexicon_ratio` (0,14 đến 0,77 trên cùng tập câu) để tô, hiện `gate` dạng số.
- **Lexicon có nhiễu**: ví dụ `với` có polarity −0,75, `trong` và `phòng` +0,75. Chỉ nên hiện `dims` trong phần mở rộng ("Vì sao?").
- **Không xử lý teencode**: lúc train nhóm không dùng bảng teencode, nên demo cũng không (để giữ đúng phân phối dữ liệu). Không chọn câu teencode ("k", "đc") làm câu mẫu.
- **Model UIT train trên phản hồi sinh viên**: câu ngoài miền (ví dụ đánh giá sản phẩm) kém tin cậy hơn; zero-shot sang VLSP2016 nhóm đo macro-F1 0,51.
- Câu có vế "nhưng": `Giảng viên dạy rất nhiệt tình nhưng phòng học hơi nóng` → **Tiêu cực 0,986** (không phải Tích cực).

## Hiệu năng đo trên máy demo (RTX 5060 Laptop 8 GB, i7-14650HX, 16 GB RAM)

- Nạp model UIT: 3,5 s; khởi động JVM VnCoreNLP: 0,5 s; câu đầu tiên ~0,7 s (khởi động CUDA, server tự chạy trước lúc khởi động).
- Mỗi câu sau đó: 7–33 ms phía server.
- Model khác nạp lần đầu khi được chọn: 1,4–3,2 s (đo được), sau đó ~10–30 ms/câu.
- Bộ nhớ: tiến trình Python ~1,4–1,8 GB RAM; VRAM ~0,5 GB mỗi model, tối đa 3 model cùng lúc.

## Kiểm chứng

- `python -m pytest demo/tests -q` (cần `JAVA_HOME`): 11 test, gồm test so logits backend với `SentimentModel.forward` (khớp tuyệt đối, cả bật và tắt lexicon), model 2 nhãn, giới hạn 3 model trong VRAM.
- Dev tính lại trên máy demo, trùng số lưu trong checkpoint: `best_fused.pt` acc 0,9501 / macro-F1 0,8685; `best.pt` 0,9558 / 0,8814; `aivivn2019_best.pt` 0,9091 / 0,9074.
