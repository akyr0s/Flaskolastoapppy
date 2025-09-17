from flask import Flask, render_template, request, send_from_directory, render_template_string, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
import logging, threading, uuid, json
import requests, base64, os, time, re
from werkzeug.utils import secure_filename

# ===== Flask / Socket.IO =====
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='eventlet')

# ===== SD API URL =====
WEBUI = os.getenv("WEBUI_URL", "http://127.0.0.1:7860")

# ===== Paths =====
SAVE_DIR = os.path.join("static", "outputs")
UPLOAD_DIR = os.path.join("static", "Uploads")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ===== Defaults / Safety =====
TARGET_CKPT = "realisticVisionV60B1_v51HyperVAE.safetensors [f47e942ad4]"
DEFAULT_SAMPLER = "DPM++ 2M SDE Karras"
DEFAULT_SAFE_NEG = (
    "nsfw, nude, naked, explicit, nipples, areola, cleavage, breasts, "
    "underboob, sideboob, lingerie, erotic, lewd, porn, sexual, "
    "suggestive, fetish, crotch, cameltoe"
)

# ===== Logger =====
logger = logging.getLogger('connection_logger')
logger.setLevel(logging.INFO)
handler = logging.FileHandler('connections.log')
handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(handler)

room_users = {"home": set(), "camera-home": set(), "camera-to-camera": set()}

def log_connection(event_type):
    from flask import request as _rq
    ip = (_rq.headers.get('X-Forwarded-For') or _rq.remote_addr or 'Unknown IP')
    ua = _rq.headers.get('User-Agent') or 'Unknown Agent'
    device = 'Mobile' if 'Mobile' in ua else 'Desktop/Other'
    logger.info(f"{event_type} | IP: {ip} | Device: {device} | UA: {ua[:50]}...")

def set_sd_options(checkpoint=None):
    if not checkpoint:
        return
    try:
        requests.post(
            f"{WEBUI}/sdapi/v1/options",
            json={"sd_model_checkpoint": checkpoint},
            timeout=15
        ).raise_for_status()
    except Exception:
        pass

# Flask 3.x: before_first_request δεν υπάρχει
_sd_init_done = False
@app.before_request
def _ensure_sd_loaded():
    global _sd_init_done
    if not _sd_init_done:
        try:
            set_sd_options(checkpoint=TARGET_CKPT)
        finally:
            _sd_init_done = True

def parse_size(val):
    try:
        w, h = val.lower().split('x')
        return int(w), int(h)
    except Exception:
        return 512, 512

# ===== ControlNet helper (νέα API: 'image', enum strings) =====
def build_controlnet_args(img_b64: str, ctype: str, preproc: str):
    if not img_b64 or not ctype:
        return {}
    ctype = ctype.lower()
    if ctype == "reference":
        module = "reference"; model = ""; preprocessor = "reference_only"
    elif ctype == "openpose":
        module = "openpose"; model = "ip-adapter-faceid-plusv2_sd15"
        preprocessor = preproc if preproc in ("openpose_full", "openpose_faceonly") else "openpose_full"
    else:
        return {}

    unit = {
        "image": f"data:image/png;base64,{img_b64}",
        "module": module,
        "model": model,
        "weight": 1.0,
        "processor_res": 512,
        "resize_mode": "Crop and Resize",  # API enum-friendly
        "guidance_start": 0.0,
        "guidance_end": 1.0,
        "control_mode": "Balanced",        # 'Balanced' | 'My prompt is more important' | 'ControlNet is more important'
        "preprocessor": preprocessor
    }
    return {"controlnet": {"args": [unit]}}

def slugify(text, maxlen=40):
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^a-zA-Z0-9_\-]+", "", text)
    return text[:maxlen] if text else "img"

def ts_name(seed):
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{ts}_seed{seed if seed!=-1 else 'RND'}"

# ===== Roop helper (σωστή σειρά args + CodeFormer) =====
def build_roop_script_args(src_b64: str,
                           face_index: str = "0",
                           model_path: str | None = None,
                           face_restorer: str = "CodeFormer",
                           restorer_visibility: float = 1.0,
                           upscaler_name: str = "None",
                           upscale_scale: int = 1,
                           upscale_visibility: float = 1.0,
                           swap_source: bool = False,
                           swap_generated: bool = True):
    """
    Συμβατό με sd-webui-roop (alwayson_scripts -> roop.args).

    Αναμενόμενη σειρά (σύμφωνα με τις πιο κοινές εκδόσεις):
      [ base64_image, enable, face_index_str, model_path,
        face_restorer, restorer_visibility,
        None,               # placeholder για 'upscaler' toggle
        upscale_scale,
        upscaler_name,
        swap_source,
        swap_generated
      ]
    """
    if not src_b64:
        return {}

    # default model path
    default_model = os.getenv(
        "ROOP_MODEL",
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)) if "__file__" in globals() else os.getcwd(),
            "models", "roop", "inswapper_128.onnx"
        )
    )
    model_path = model_path or default_model

    roop_args = [
        f"data:image/png;base64,{src_b64}",  # Base64 of face image
        True,                                # enable
        str(face_index or "0"),              # face index string
        model_path,                          # model path
        face_restorer,                       # 'CodeFormer' | 'GFPGAN' | 'None'
        float(restorer_visibility),          # 0..1
        None,                                # placeholder (upscaler toggle not used)
        int(upscale_scale),                  # 1..n
        upscaler_name,                       # 'None' ή όνομα upscaler
        bool(swap_source),                   # swap in source
        bool(swap_generated)                 # swap in generated
    ]

    return {"roop": {"args": roop_args}}

# ===== Pages =====
@app.route('/')
def home(): return render_template('home.html')

@app.route('/camera-home')
def camera_home(): return render_template('camera_home.html')

@app.route('/camera-to-camera')
def camera_to_camera(): return render_template('camera_to_camera.html')

@app.route('/landing')
def landing(): return send_from_directory('static', 'landing.html')

@app.route('/chat')
def chat(): return render_template('chat.html')

@app.route('/moderator')
def moderator(): return render_template('moderator.html')

@app.route('/cartoon')
def cartoon(): return render_template('cartoon.html')

# ===== SD UI (inline) =====
@app.route('/sd')
def sd_home():
    ctx = {"model_label": "RealisticVisionV60B1_v51HyperVAE", "img_b64": None}
    return render_template_string(SD_HTML, **ctx)

def _read_file_b64(fs_file, prefix_name, target_dir):
    if not fs_file or not getattr(fs_file, "filename", ""):
        return None, None
    tsu = int(time.time())
    safe = secure_filename(fs_file.filename or f"{prefix_name}.png")
    path = os.path.join(target_dir, f"{tsu}_{safe}")
    fs_file.save(path)
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode(), path

# ===== txt2img =====
@app.post('/sd/txt2img')
def sd_txt2img():
    prompt = request.form.get('prompt','')
    neg = request.form.get('neg','').strip() or DEFAULT_SAFE_NEG
    steps = int(request.form.get('steps', 20))
    cfg = float(request.form.get('cfg', 7))
    size = request.form.get('size','512x512')
    w, h = parse_size(size)
    n_iter = max(1, min(4, int(request.form.get('batch', 1))))
    seed = int(request.form.get('seed', -1))

    roop_b64, _ = _read_file_b64(request.files.get('roop_source'), "source", UPLOAD_DIR)
    cn_b64, _ = _read_file_b64(request.files.get('control_image'), "control", UPLOAD_DIR)
    control_type = request.form.get('control_type','')
    control_pre  = request.form.get('control_pre','')

    payload = {
        "prompt": prompt,
        "negative_prompt": neg,
        "steps": steps,
        "cfg_scale": cfg,
        "width": w,
        "height": h,
        "n_iter": n_iter,
        "seed": seed,
        "sampler_name": DEFAULT_SAMPLER
    }

    # Hires
    hires_enable = request.form.get('hires_enable') == 'on'
    hires_scale = float(request.form.get('hires_scale', 1.0))
    hires_upscaler = request.form.get('hires_upscaler', 'R-ESRGAN 4x+')
    if hires_enable and hires_scale > 1.0:
        payload.update({
            "enable_hr": True,
            "hr_scale": min(3.0, max(1.0, hires_scale)),
            "hr_upscaler": hires_upscaler,
            "hr_second_pass_steps": max(0, int(steps//2))
        })

    # Always-on
    always = {}
    if roop_b64:
        always.update(build_roop_script_args(
            roop_b64,
            face_index=request.form.get("roop_face_index","0"),
            model_path=request.form.get("roop_model",""),
            face_restorer="CodeFormer",
            restorer_visibility=float(request.form.get("roop_restorer_vis","1")),
            upscaler_name=request.form.get("roop_upscaler","None"),
            upscale_scale=int(request.form.get("roop_upscale_scale","1")),
            upscale_visibility=float(request.form.get("roop_upscale_vis","1")),
            swap_source=(request.form.get("roop_swap_source") == "on"),
            swap_generated=True
        ))
    if cn_b64:
        always.update(build_controlnet_args(cn_b64, control_type, control_pre))
    if always:
        payload["alwayson_scripts"] = always

    r = requests.post(f"{WEBUI}/sdapi/v1/txt2img", json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    img_b64 = data["images"][0]

    seed_used = seed
    try:
        info = json.loads(data.get("info","{}")); seed_used = info.get("seed", seed)
    except Exception:
        pass

    fname = f"{ts_name(seed_used)}_{slugify(prompt)}.png"
    out_path = os.path.join(SAVE_DIR, fname)
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(img_b64))

    return jsonify(ok=True, img_b64=img_b64, seed=seed_used)

# ===== img2img =====
@app.post('/sd/img2img')
def sd_img2img():
    in_file = request.files['image']
    b64, in_path = _read_file_b64(in_file, "input", UPLOAD_DIR)

    prompt = request.form.get('prompt','')
    neg = request.form.get('neg','').strip() or DEFAULT_SAFE_NEG
    denoise = float(request.form.get('denoise', 0.45))
    steps = int(request.form.get('steps', 20))
    cfg = float(request.form.get('cfg', 7))
    n_iter = max(1, min(4, int(request.form.get('batch', 1))))
    seed = int(request.form.get('seed', -1))

    roop_b64, _ = _read_file_b64(request.files.get('roop_source'), "source", UPLOAD_DIR)
    cn_b64, _ = _read_file_b64(request.files.get('control_image'), "control", UPLOAD_DIR)
    control_type = request.form.get('control_type','')
    control_pre  = request.form.get('control_pre','')

    payload = {
        "prompt": prompt,
        "negative_prompt": neg,
        "denoising_strength": denoise,
        "init_images": [b64],
        "steps": steps,
        "cfg_scale": cfg,
        "n_iter": n_iter,
        "seed": seed,
        "sampler_name": DEFAULT_SAMPLER
    }

    # Hires
    hires_enable = request.form.get('hires_enable') == 'on'
    hires_scale = float(request.form.get('hires_scale', 1.0))
    hires_upscaler = request.form.get('hires_upscaler', 'R-ESRGAN 4x+')
    if hires_enable and hires_scale > 1.0:
        payload.update({
            "enable_hr": True,
            "hr_scale": min(3.0, max(1.0, hires_scale)),
            "hr_upscaler": hires_upscaler,
            "hr_second_pass_steps": max(0, int(steps//2))
        })

    always = {}
    if roop_b64:
        always.update(build_roop_script_args(
            roop_b64,
            face_index=request.form.get("roop_face_index","0"),
            model_path=request.form.get("roop_model",""),
            face_restorer="CodeFormer",
            restorer_visibility=float(request.form.get("roop_restorer_vis","1")),
            upscaler_name=request.form.get("roop_upscaler","None"),
            upscale_scale=int(request.form.get("roop_upscale_scale","1")),
            upscale_visibility=float(request.form.get("roop_upscale_vis","1")),
            swap_source=(request.form.get("roop_swap_source") == "on"),
            swap_generated=True
        ))
    if cn_b64:
        always.update(build_controlnet_args(cn_b64, control_type, control_pre))
    if always:
        payload["alwayson_scripts"] = always

    r = requests.post(f"{WEBUI}/sdapi/v1/img2img", json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    img_b64 = data["images"][0]

    seed_used = seed
    try:
        info = json.loads(data.get("info","{}")); seed_used = info.get("seed", seed)
    except Exception:
        pass

    fname = f"{ts_name(seed_used)}_{slugify(prompt)}.png"
    out_path = os.path.join(SAVE_DIR, fname)
    with open(out_path, "wb") as fo:
        fo.write(base64.b64decode(img_b64))

    return jsonify(ok=True, img_b64=img_b64, seed=seed_used)

# ===== Gallery =====
@app.get("/sd/gallery")
def sd_gallery():
    files = [fn for fn in sorted(os.listdir(SAVE_DIR), reverse=True) if fn.lower().endswith(".png")][:400]
    items = "".join(
        f'<a href="/{SAVE_DIR}/{fn}" target="_blank"><img src="/{SAVE_DIR}/{fn}" loading="lazy"></a>'
        for fn in files
    )
    return f"""
<!doctype html>
<html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gallery</title>
<style>
:root{{color-scheme:dark}}
body{{background:#0a0a0a;color:#e5e5e5;margin:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,"Helvetica Neue",sans-serif}}
.wrap{{max-width:1200px;margin:0 auto;padding:8px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:12px;align-items:start}}
.grid a{{display:block}}
.grid img{{width:100%;height:auto;border-radius:8px;border:1px solid #222;display:block}}
</style>
</head><body><div class="wrap"><div class="grid">{items}</div></div></body></html>
"""

# ===== Sockets (όπως ήταν) =====
waiting_user = None
chat_rooms = {}
moderator_sid = None

@socketio.on('connect')
def on_connect(): log_connection('Connect')

@socketio.on('disconnect')
def on_disconnect():
    log_connection('Disconnect')
    from flask import request as _rq
    sid = _rq.sid
    global waiting_user, moderator_sid
    if waiting_user == sid: waiting_user = None
    if moderator_sid == sid: moderator_sid = None
    for room, users in list(chat_rooms.items()):
        if sid in users:
            users.remove(sid)
            for u in users: emit('partner-left', to=u)
            del chat_rooms[room]; break
    for room, users in room_users.items():
        if sid in users:
            users.remove(sid)
            emit('users_count', {'count': len(users)}, room=room)
            break

@socketio.on('join')
def on_join(data):
    from flask import request as _rq
    room = data.get('room'); sid = _rq.sid
    join_room(room)
    if room not in room_users: room_users[room] = set()
    room_users[room].add(sid)
    emit('users_count', {'count': len(room_users[room])}, room=room)
    emit('user-joined', {'room': room}, room=room, include_self=False)

@socketio.on('join-chat')
def handle_join_chat():
    from flask import request as _rq
    global waiting_user
    sid = _rq.sid
    if waiting_user is None:
        waiting_user = sid
    else:
        room_id = str(uuid.uuid4())
        chat_rooms[room_id] = [waiting_user, sid]
        emit('match-found', {'room': room_id}, to=waiting_user)
        emit('match-found', {'room': room_id}, to=sid)
        waiting_user = None
        if moderator_sid: emit('moderator-watch', {'room': room_id}, to=moderator_sid)

@socketio.on('signal')
def on_signal(data):
    from flask import request as _rq
    room = data.get('room')
    if room in chat_rooms:
        for sid in chat_rooms.get(room, []):
            if sid != _rq.sid: emit('signal', data, to=sid)
        if moderator_sid: emit('signal', data, to=moderator_sid)
    else:
        emit('signal', data, room=room, include_self=False)

@socketio.on('moderator-join')
def handle_moderator():
    from flask import request as _rq
    global moderator_sid
    moderator_sid = _rq.sid

# ===== Inline minimal SD UI (ίδιο) =====
SD_HTML = r"""
<!doctype html>
<html lang="el" class="h-full">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Makis PLAYground</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>#loading{pointer-events:all}</style>
</head>
<body class="h-full bg-neutral-950 text-neutral-100">
  <div class="max-w-6xl mx-auto p-4 md:p-6">
    <header class="flex items-center justify-between mb-4">
      <h1 class="text-xl md:text-2xl font-semibold">Makis PLAYground</h1>
      <span class="text-xs opacity-70 -mt-1">{{model_label}}</span>
    </header>

    <div class="flex gap-2 mb-4">
      <button class="tab-btn px-3 py-2 rounded-lg bg-neutral-800" data-tab="txt2img">Δημιουργία κειμένου σε εικόνα</button>
      <button class="tab-btn px-3 py-2 rounded-lg bg-neutral-900" data-tab="img2img">Δημιουργία εικόνας σε εικόνα</button>
      <button class="tab-btn px-3 py-2 rounded-lg bg-neutral-900" data-tab="gallery">Gallery</button>
    </div>

    <!-- TXT2IMG -->
    <section id="panel-txt2img" class="grid md:grid-cols-2 gap-6">
      <form id="form-t2i" class="space-y-4" enctype="multipart/form-data">
        <button type="submit" class="btn-submit w-full md:w-auto px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500">Generate</button>

        <div>
          <div class="flex items-center justify-between">
            <label class="text-sm opacity-80">Prompt</label>
            <button type="button" class="px-3 py-1 rounded bg-neutral-800 text-xs" id="randPromptT2I">Τυχαίο</button>
          </div>
          <textarea id="prompt_t2i" name="prompt" rows="3" class="w-full mt-1 p-3 rounded-lg bg-neutral-900 border border-neutral-800"></textarea>
        </div>

        <div>
          <div class="flex items-center justify-between">
            <label class="text-sm opacity-80">Negative</label>
            <button type="button" class="px-3 py-1 rounded bg-neutral-800 text-xs" id="randNegT2I">Τυχαίο</button>
          </div>
          <input id="neg_t2i" name="neg" class="w-full mt-1 p-3 rounded-lg bg-neutral-900 border border-neutral-800"/>
        </div>

        <!-- Roop -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-3 border border-neutral-800 rounded-lg p-3">
          <div>
            <label class="text-sm opacity-80">Roop source</label>
            <input type="file" name="roop_source" accept="image/*" class="w-full p-2 rounded-lg bg-neutral-900 border border-neutral-800" id="roop_t2i">
          </div>
          <div class="text-xs opacity-70">
            Face-swap με CodeFormer. Χρήση μοντέλου <code>models/roop/inswapper_128.onnx</code>.
          </div>
        </div>

        <!-- ControlNet -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-3 border border-neutral-800 rounded-lg p-3">
          <div>
            <label class="text-sm opacity-80">ControlNet image</label>
            <input type="file" name="control_image" accept="image/*" class="w-full p-2 rounded-lg bg-neutral-900 border border-neutral-800" id="ctrl_t2i">
          </div>
          <div class="grid grid-cols-2 gap-2 items-start">
            <div>
              <label class="text-xs opacity-80">Control Type</label>
              <select name="control_type" id="ctype_t2i" class="w-full p-2 rounded-lg bg-neutral-900 border border-neutral-800">
                <option value="">—</option>
                <option value="reference">Reference (reference_only)</option>
                <option value="openpose">OpenPose</option>
              </select>
            </div>
            <div id="cpre_wrap_t2i" class="hidden">
              <label class="text-xs opacity-80">Preprocessor</label>
              <select name="control_pre" id="cpre_t2i" class="w-full p-2 rounded-lg bg-neutral-900 border border-neutral-800">
                <option value="openpose_full">openpose_full</option>
                <option value="openpose_faceonly">openpose_faceonly</option>
              </select>
            </div>
          </div>
        </div>

        <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div>
            <label class="text-sm opacity-80">Steps</label>
            <select name="steps" class="w-full mt-1 p-2 rounded-lg bg-neutral-900 border border-neutral-800">
              {% for n in range(5,31) %}<option {% if n==20 %}selected{% endif %}>{{n}}</option>{% endfor %}
            </select>
          </div>
          <div>
            <label class="text-sm opacity-80">CFG</label>
            <select name="cfg" class="w-full mt-1 p-2 rounded-lg bg-neutral-900 border border-neutral-800">
              {% for n in range(1,16) %}<option {% if n==7 %}selected{% endif %}>{{n}}</option>{% endfor %}
            </select>
          </div>
          <div>
            <label class="text-sm opacity-80">Seed</label>
            <input id="seed_t2i" name="seed" type="number" value="-1" class="w-full mt-1 p-2 rounded-lg bg-neutral-900 border border-neutral-800"/>
          </div>
          <div>
            <label class="text-sm opacity-80">Batch (1–4)</label>
            <input name="batch" type="number" min="1" max="4" value="1" class="w-full mt-1 p-2 rounded-lg bg-neutral-900 border border-neutral-800"/>
          </div>
        </div>

        <div class="grid grid-cols-2 md:grid-cols-3 gap-3">
          <div class="col-span-2 md:col-span-1">
            <label class="text-sm opacity-80">Μέγεθος</label>
            <select name="size" class="w-full mt-1 p-2 rounded-lg bg-neutral-900 border border-neutral-800">
              <option selected>512x512</option>
              <option>768x768</option>
              <option>960x960</option>
              <option>1024x768</option>
              <option>1024x1024</option>
            </select>
          </div>
          <div class="md:col-span-2">
            <label class="text-sm opacity-80">Hires fix</label>
            <div class="flex items-center gap-2 mt-1">
              <input type="checkbox" name="hires_enable">
              <select name="hires_upscaler" class="p-2 rounded-lg bg-neutral-900 border border-neutral-800">
                <option>Latent</option>
                <option>Latent (antialiased)</option>
                <option selected>R-ESRGAN 4x+</option>
              </select>
              <select name="hires_scale" class="w-24 p-2 rounded-lg bg-neutral-900 border border-neutral-800">
                <option selected>1</option><option>2</option><option>3</option>
              </select>
            </div>
          </div>
        </div>
      </form>

      <div>
        <div class="text-xs opacity-70 mb-2">Outputs: <code>static/outputs/</code></div>
        <div class="text-sm opacity-70">Καμία εικόνα ακόμη</div>
      </div>
    </section>

    <!-- IMG2IMG -->
    <section id="panel-img2img" class="hidden grid md:grid-cols-2 gap-6">
      <form id="form-i2i" class="space-y-4" enctype="multipart/form-data">
        <button type="submit" class="btn-submit w-full md:w-auto px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500">Transform</button>

        <div>
          <label class="text-sm opacity-80">Εικόνα</label>
          <input type="file" name="image" accept="image/*" required class="w-full mt-1 p-2 rounded-lg bg-neutral-900 border border-neutral-800" id="i2i_image">
        </div>

        <div>
          <div class="flex items-center justify-between">
            <label class="text-sm opacity-80">Prompt</label>
            <button type="button" class="px-3 py-1 rounded bg-neutral-800 text-xs" id="randPromptI2I">Τυχαίο</button>
          </div>
          <textarea id="prompt_i2i" name="prompt" rows="3" class="w-full mt-1 p-3 rounded-lg bg-neutral-900 border border-neutral-800"></textarea>
        </div>

        <div>
          <div class="flex items-center justify-between">
            <label class="text-sm opacity-80">Negative</label>
            <button type="button" class="px-3 py-1 rounded bg-neutral-800 text-xs" id="randNegI2I">Τυχαίο</button>
          </div>
          <input id="neg_i2i" name="neg" class="w-full mt-1 p-3 rounded-lg bg-neutral-900 border border-neutral-800"/>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-3 border border-neutral-800 rounded-lg p-3">
          <div>
            <label class="text-sm opacity-80">Roop source</label>
            <input type="file" name="roop_source" accept="image/*" class="w-full p-2 rounded-lg bg-neutral-900 border border-neutral-800" id="roop_i2i">
          </div>
          <div class="text-xs opacity-70">Face-swap πάνω στην target εικόνα.</div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-3 border border-neutral-800 rounded-lg p-3">
          <div>
            <label class="text-sm opacity-80">ControlNet image</label>
            <input type="file" name="control_image" accept="image/*" class="w-full p-2 rounded-lg bg-neutral-900 border border-neutral-800" id="ctrl_i2i">
          </div>
          <div class="grid grid-cols-2 gap-2 items-start">
            <div>
              <label class="text-xs opacity-80">Control Type</label>
              <select name="control_type" id="ctype_i2i" class="w-full p-2 rounded-lg bg-neutral-900 border border-neutral-800">
                <option value="">—</option>
                <option value="reference">Reference (reference_only)</option>
                <option value="openpose">OpenPose</option>
              </select>
            </div>
            <div id="cpre_wrap_i2i" class="hidden">
              <label class="text-xs opacity-80">Preprocessor</label>
              <select name="control_pre" id="cpre_i2i" class="w-full p-2 rounded-lg bg-neutral-900 border border-neutral-800">
                <option value="openpose_full">openpose_full</option>
                <option value="openpose_faceonly">openpose_faceonly</option>
              </select>
            </div>
          </div>
        </div>

        <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div>
            <label class="text-sm opacity-80">Denoise</label>
            <input name="denoise" type="number" step="0.05" min="0" max="1" value="0.45" class="w-full mt-1 p-2 rounded-lg bg-neutral-900 border border-neutral-800"/>
          </div>
          <div>
            <label class="text-sm opacity-80">Steps</label>
            <select name="steps" class="w-full mt-1 p-2 rounded-lg bg-neutral-900 border border-neutral-800">
              {% for n in range(5,31) %}<option {% if n==20 %}selected{% endif %}>{{n}}</option>{% endfor %}
            </select>
          </div>
          <div>
            <label class="text-sm opacity-80">CFG</label>
            <select name="cfg" class="w-full mt-1 p-2 rounded-lg bg-neutral-900 border border-neutral-800">
              {% for n in range(1,16) %}<option {% if n==7 %}selected{% endif %}>{{n}}</option>{% endfor %}
            </select>
          </div>
          <div>
            <label class="text-sm opacity-80">Seed</label>
            <input id="seed_i2i" name="seed" type="number" value="-1" class="w-full mt-1 p-2 rounded-lg bg-neutral-900 border border-neutral-800"/>
          </div>
        </div>

        <div>
          <label class="text-sm opacity-80">Batch (1–4)</label>
          <input name="batch" type="number" min="1" max="4" value="1" class="w-full mt-1 p-2 rounded-lg bg-neutral-900 border border-neutral-800"/>
        </div>

        <div>
          <label class="text-sm opacity-80">Hires fix</label>
          <div class="flex items-center gap-2 mt-1">
            <input type="checkbox" name="hires_enable">
            <select name="hires_upscaler" class="p-2 rounded-lg bg-neutral-900 border border-neutral-800">
              <option>Latent</option>
              <option>Latent (antialiased)</option>
              <option selected>R-ESRGAN 4x+</option>
            </select>
            <select name="hires_scale" class="w-24 p-2 rounded-lg bg-neutral-900 border border-neutral-800">
              <option selected>1</option><option>2</option><option>3</option>
            </select>
          </div>
        </div>
      </form>

      <div>
        <div class="text-xs opacity-70 mb-2">Outputs: <code>static/outputs/</code></div>
        <div class="text-sm opacity-70">Καμία εικόνα ακόμη</div>
      </div>
    </section>
  </div>

  <div id="loading" class="fixed inset-0 hidden items-center justify-center bg-black/60 z-50">
    <div class="px-5 py-3 rounded-lg bg-neutral-900 border border-neutral-700 text-sm">Παρακαλώ περιμένετε…</div>
  </div>

  <script>
    const RAND_PROMPTS=["ultra-detailed portrait of a young woman, cinematic lighting, 85mm lens, f/1.4 aperture, shallow depth of field, hyperrealistic skin texture, subtle makeup, bokeh background",
  "dramatic black and white portrait of an old man, Rembrandt lighting, deep wrinkles, high contrast shadows, Leica 50mm Summilux",
  "studio headshot of a confident businessman, seamless grey backdrop, professional lighting setup, ultra sharp focus",
  "portrait of a red-haired woman outdoors, golden hour sunlight, wind in her hair, soft rim lighting, Canon 135mm f/2",
  "intimate close-up of a smiling woman, freckles, natural daylight, Nikon Z9 85mm lens, authentic expression",
  "cinematic portrait of a bearded man in a leather jacket, moody tones, neon reflections in glasses",
  "fashion editorial portrait of a model in avant-garde outfit, high-end studio lighting, Vogue photography style",
  "dreamy portrait of a girl with curly hair, soft pastel background, Fujifilm GFX, airy and ethereal tone",
  "artistic portrait of a woman with colorful paint strokes on her face, surreal composition, Hasselblad clarity",
  "urban portrait of a man with hoodie, cinematic teal and orange color grade, Sony A7R V, dramatic lighting",
  "bridal portrait of a woman in wedding dress, glowing atmosphere, airy bokeh, 50mm lens at f/1.2",
  "dark noir portrait of a man smoking, harsh shadows, chiaroscuro, cinematic mood",
  "portrait of a child laughing, sunlight bokeh in background, Canon RF 85mm, vibrant natural tones",
  "minimalist portrait of a woman with short haircut, hard light contrast, bold shadows",
  "street portrait of a man in leather jacket, gritty vibe, shallow DOF, Leica M11",
  "classic oil-painting style portrait of an elegant lady with pearl earrings, realistic digital painting",
  "portrait of a tattooed hipster man in a coffee shop, warm ambient light, lifestyle photography style",
  "cinematic portrait of a young woman with long dark hair, mysterious vibes, anamorphic lens",
  "stage portrait of a ballerina mid-pose, dramatic spotlight, graceful movement, wide aperture",
  "ethereal portrait of a woman with flowers in her hair, whimsical styling, 85mm lens, pastel tones",
  "close-up of intense male eyes, dramatic chiaroscuro lighting, textured skin, ultra-detailed",
  "joyful portrait of an elderly woman smiling, wrinkles detailed, warm golden sunlight, authenticity",
  "portrait of a boy in traditional cultural clothing, natural daylight, storytelling composition",
  "vintage portrait of a mysterious woman with wide-brimmed hat, cinematic tones, soft focus",
  "rockstar portrait with guitar, edgy lighting, concert atmosphere, sharp details",
  "scientist in lab coat portrait, sterile background, cinematic depth, ultra clarity",
  "author portrait in library, moody tones, shallow DOF, Sony A9",
  "hollywood glamour portrait, soft key light, glossy makeup, cinematic lens flares",
  "military portrait of a soldier, gritty tone, cinematic film grain, dramatic shadows",
  "cyberpunk portrait of a woman with neon makeup, glowing reflections, futuristic atmosphere",
  "executive portrait of a man in suit, clean white background, ultra sharp, corporate look",
  "gothic portrait of a woman in dark lace dress, candlelight, cinematic shadows",
  "farmer portrait outdoors, natural warm light, storytelling rural vibe",
  "cozy portrait of a girl holding a cat, authentic expression, soft window light",
  "DJ portrait, man with headphones, club neon lights, cinematic composition",
  "nurse portrait in hospital, clinical lighting, professional headshot",
  "artistic portrait of a woman with rainbow-colored hair, neon lighting setup, edgy style",
  "athlete portrait in gym, dramatic lighting on muscles, sweat details, ultra-sharp",
  "teacher portrait in classroom, soft window light, approachable vibe",
  "chef portrait in kitchen, cinematic depth, warm tungsten tones",
  "macro portrait of a woman’s eye with dramatic makeup, hyperrealism",
  "rainy portrait of a man in trench coat, cinematic rain reflections, moody tone",
  "editorial couple portrait, high-fashion lighting, cinematic Vogue aesthetic",
  "lawyer portrait in office, confident expression, controlled lighting",
  "monk portrait by candlelight, serene expression, chiaroscuro shadows",
  "pilot portrait in airplane cockpit, cinematic lighting, storytelling details",
  "traveler portrait with backpack, mountains in background, wide aperture",
  "family portrait outdoors, soft golden hour, warm tones, authentic smiles",
  "graceful ballerina portrait, spotlight stage, elegant pose, cinematic contrast"];
    const RAND_NEGS=["nsfw, nude, naked, lowres, blurry, deformed, distorted, bad anatomy, cropped, out of frame, text, watermark, logo, signature, extra fingers, extra limbs, artifacts",
  "low quality, jpeg artifacts, noise, oversaturated, underexposed, overexposed, unnatural skin tones, fake look, cartoonish, mutated, glitch",
  "unnatural hands, malformed arms, twisted neck, missing limbs, cloned face, duplicate head, unnatural pose",
  "bad composition, low detail, flat lighting, poor contrast, low dynamic range, unrealistic proportions",
  "extra body parts, multiple heads, asymmetrical face, lopsided eyes, deformed mouth, bad teeth",
  "text overlay, UI elements, border, frame, collage, screenshot look, computer UI artifacts",
  "color banding, compression artifacts, pixelated, mosaic, aliasing",
  "mutated skin, rough texture, bad shading, oversharpened, unappealing look",
  "AI artifacts, blurry background, unrealistic depth of field, fake bokeh, duplicated objects",
  "extra ears, missing nose, distorted jawline, broken symmetry, asymmetry, stretched features",
  "unrealistic lighting, overexposed highlights, crushed blacks, unnatural shadows",
  "distorted proportions, wide face, flattened head, wrong perspective",
  "clipping, broken edges, cropped forehead, missing top of head",
  "ugly, unpleasant face, poorly drawn, low fidelity, messy rendering",
  "wrong clothing details, incorrect textures, random patterns",
  "oversized eyes, melted face, fused skin, blurry mouth",
  "unnatural reflections in glasses, fake shine, broken transparency",
  "distracting background, random objects, irrelevant scenery",
  "wrong number of fingers, elongated hands, melted anatomy",
  "unnatural facial expression, awkward smile, uncanny valley effect",
  "bad hair details, blurry strands, pixelated hair, fused hair blocks",
  "repetitive patterns, duplicate features, tiling effect, cloning artifacts",
  "camera watermark, brand logo, text overlay, noise pattern",
  "unwanted blur, camera shake effect, unsharp mask gone wrong",
  "wrong ethnicity details, generic AI face, plastic look",
  "flat textures, bad subsurface scattering, waxy skin",
  "posterization, color mismatch, hue shifts, neon oversaturation",
  "weird hand gestures, unrealistic arms, broken elbow joint",
  "fake reflections, unrealistic glass, distorted perspective",
  "weird body proportions, torso too long, neck too thick"];
    function pick(a){return a[Math.floor(Math.random()*a.length)]}
    function bindRandom(btnId, inputId){document.getElementById(btnId)?.addEventListener('click',()=>{document.getElementById(inputId).value=btnId.toLowerCase().includes('neg')?pick(RAND_NEGS):pick(RAND_PROMPTS)})}

    const tabs=document.querySelectorAll('.tab-btn');
    const panels={txt2img:document.getElementById('panel-txt2img'),img2img:document.getElementById('panel-img2img')};
    function activateTab(key){
      if(key==='gallery'){window.open('/sd/gallery','_blank');return;}
      for(const k in panels){panels[k].classList.toggle('hidden',k!==key)}
      tabs.forEach(b=>b.classList.remove('bg-neutral-800'));
      document.querySelector(`.tab-btn[data-tab="${key}"]`)?.classList.add('bg-neutral-800');
    }
    tabs.forEach(btn=>btn.addEventListener('click',()=>activateTab(btn.dataset.tab)));
    activateTab('txt2img');

    const loading=document.getElementById('loading');
    async function ajaxSubmit(form, endpoint){
      const fd=new FormData(form);
      try{
        const res=await fetch(endpoint,{method:'POST',body:fd});
        const data=await res.json();
        if(data.ok&&data.img_b64){ showResult(data.img_b64); window.lastSeed=data.seed; }
        else{ alert('Σφάλμα δημιουργίας εικόνας.'); }
      }catch(e){ alert('Network/Server error.'); }
      finally{
        loading.classList.add('hidden');loading.classList.remove('flex');
        form.querySelectorAll('button[type="submit"]').forEach(el=>el.disabled=false);
      }
    }
    function attachSubmit(form, endpoint){
      form?.addEventListener('submit',(e)=>{
        e.preventDefault();
        loading.classList.remove('hidden');loading.classList.add('flex');
        form.querySelectorAll('button[type="submit"]').forEach(el=>el.disabled=true);
        ajaxSubmit(form, endpoint);
      });
    }
    function showResult(b64){
      const w=window.open("");
      w.document.write('<img style="max-width:100%" src="data:image/png;base64,'+b64+'">');
    }
    attachSubmit(document.getElementById('form-t2i'), '/sd/txt2img');
    attachSubmit(document.getElementById('form-i2i'), '/sd/img2img');

    function bindCType(selId, wrapId){
      const sel=document.getElementById(selId);
      const wrap=document.getElementById(wrapId);
      sel?.addEventListener('change',()=>{wrap.classList.toggle('hidden', sel.value!=='openpose')});
    }
    bindCType('ctype_t2i','cpre_wrap_t2i');
    bindCType('ctype_i2i','cpre_wrap_i2i');

    function bindRandom(btnId, inputId){
      document.getElementById(btnId)?.addEventListener('click',()=>{document.getElementById(inputId).value=btnId.toLowerCase().includes('neg')?pick(RAND_NEGS):pick(RAND_PROMPTS)})
    }
    bindRandom('randPromptT2I','prompt_t2i');
    bindRandom('randNegT2I','neg_t2i');
    bindRandom('randPromptI2I','prompt_i2i');
    bindRandom('randNegI2I','neg_i2i');
  </script>
</body>
</html>
"""

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
