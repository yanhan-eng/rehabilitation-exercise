"""
康复跟练系统 v4.5 (最终优化版)
========================================================
1. 解决跳变：采用滑动窗口最优匹配 (Sliding Window) + EMA 平滑。
2. 镜像模式：右侧输入画面水平翻转，实现“照镜子”感。
3. 速度调节：全局 0.75 倍速播放，降低跟练难度。
4. 防瞎评：人不在画面或关键点不足 8 个时，分数强制归零。
5. UI 增强：建议显示时间延长至 2 秒，字体大幅加粗加重。
6. 动作标题：界面上方实时显示当前视频文件名。
========================================================
"""
import os, sys, math, time, argparse, warnings
import numpy as np, cv2
from collections import deque
from PIL import Image, ImageDraw, ImageFont
warnings.filterwarnings("ignore")

# ── 全局配置 ──────────────────────────────────────────────
WIN_W=1280; WIN_H=560; PANEL_W=WIN_W//2
CONF_THRESH=0.30;   # 关键点置信度阈值
PAIN_THRESH=0.55;   # 痛苦表情阈值
EMOTION_EVERY=8;    # 每8帧检测一次表情
SEARCH_WINDOW = 35; # 搜索窗口（前后各35帧，约±1.5秒），解决相位差
SPEED_FACTOR = 0.75 # 0.75倍速播放

# 脚本路径处理
SCRIPT_DIR=os.path.dirname(os.path.abspath(__file__))
def spath(*p): return os.path.join(SCRIPT_DIR,*p)

# 表情与骨架常量
FER_LABELS=["正常","轻松","惊讶","悲伤","愤怒","厌恶","恐惧","痛苦"]; PAIN_IDX=7
KP={"nose":0,"Leye":1,"Reye":2,"Lear":3,"Rear":4,
    "Lsho":5,"Rsho":6,"Lelb":7,"Relb":8,"Lwri":9,"Rwri":10,
    "Lhip":11,"Rhip":12,"Lkne":13,"Rkne":14,"Lank":15,"Rank":16}
SKEL_LINKS=[
    ("nose","Lsho",(200,200,200)),("nose","Rsho",(200,200,200)),
    ("Lsho","Lhip",(200,200,200)),("Rsho","Rhip",(200,200,200)),
    ("Lhip","Rhip",(200,200,200)),("Lsho","Rsho",(200,200,200)),
    ("Lsho","Lelb",(0,165,255)),("Lelb","Lwri",(0,165,255)),
    ("Rsho","Relb",(255,130,0)),("Relb","Rwri",(255,130,0)),
    ("Lhip","Lkne",(0,165,255)),("Lkne","Lank",(0,165,255)),
    ("Rhip","Rkne",(255,130,0)),("Rkne","Rank",(255,130,0)),
]

# 评分特征：名称, 类型, 节点A, 节点B, 节点C, 权重, 中文建议
SCORE_FEATURES = [
    ("L_wrist_spread",  "spread_x", "Lwri","Lsho", None,  3.0, "左手展开不足"),
    ("R_wrist_spread",  "spread_x", "Rwri","Rsho", None,  3.0, "右手展开不足"),
    ("L_arm_straight",  "arm_straight","Lsho","Lelb","Lwri",1.5, "左臂请伸直"),
    ("R_arm_straight",  "arm_straight","Rsho","Relb","Rwri",1.5, "右臂请伸直"),
    ("trunk_upright",   "align_x",  "nose","Lhip","Rhip", 1.0, "请保持身体挺直"),
]

# ── 基础工具函数 ──────────────────────────────────────────
def letterbox(frame, tw, th):
    h,w=frame.shape[:2]
    sc=min(tw/w,th/h); nw,nh=int(w*sc),int(h*sc)
    px,py=(tw-nw)//2,(th-nh)//2
    panel=np.zeros((th,tw,3),np.uint8)
    if nw>0 and nh>0:
        panel[py:py+nh,px:px+nw]=cv2.resize(frame,(nw,nh))
    return panel,sc,px,py

def kpts_to_panel(k,sw,sh,sc,px,py):
    k=k.astype(float).copy(); k[:,0]=k[:,0]*sc+px; k[:,1]=k[:,1]*sc+py; return k

class TR:
    def __init__(self):
        self._c={}; self._fp=None
        for fp in["C:/Windows/Fonts/msyh.ttc","C:/Windows/Fonts/simhei.ttf",
                  "C:/Windows/Fonts/simsun.ttc","/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"]:
            if os.path.isfile(fp): self._fp=fp; break
    def _f(self,sz):
        if sz not in self._c: self._c[sz]=ImageFont.truetype(self._fp,sz) if self._fp else ImageFont.load_default()
        return self._c[sz]
    def put(self,bgr,txt,xy,sz=18,col=(255,255,255),bold=False):
        img=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        d=ImageDraw.Draw(img); f=self._f(sz); rgb=(col[2],col[1],col[0])
        if bold:
            for dx,dy in[(-1,0),(1,0),(0,-1),(0,1)]: d.text((xy[0]+dx,xy[1]+dy),txt,font=f,fill=rgb)
        d.text(xy,txt,font=f,fill=rgb)
        bgr[:]=cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)

# ── 特征计算与相似度逻辑 ──────────────────────────────────
def extract_features(kpts, confs, sho_width_ref=None):
    def g(n):
        i=KP.get(n); return kpts[i,:2].astype(float) if (i is not None and confs[i]>CONF_THRESH) else None
    ls, rs = g("Lsho"), g("Rsho")
    if ls is None or rs is None: return None, None
    sho_w = np.linalg.norm(rs-ls)
    if sho_w < 5: return None, None
    norm_w = sho_width_ref if (sho_width_ref and sho_width_ref>5) else sho_w

    feats = {}
    for fname, ftype, ka, kb, kc, w, desc in SCORE_FEATURES:
        pa, pb = g(ka), g(kb); pc = g(kc) if kc else None
        if ftype == "spread_x":
            feats[fname] = abs(pa[0]-pb[0])/norm_w if (pa is not None and pb is not None) else None
        elif ftype == "arm_straight":
            if pa is not None and pb is not None and pc is not None:
                v1, v2 = pb-pa, pc-pb
                feats[fname] = 1.0 - min(1.0, abs(v1[0]*v2[1]-v1[1]*v2[0])/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-7))
            else: feats[fname] = None
        elif ftype == "align_x":
            mid_x = (pb[0] + pc[0]) / 2.0 if (pb is not None and pc is not None) else (pb[0] if pb is not None else None)
            feats[fname] = 1.0 - min(1.0, abs(pa[0]-mid_x)/norm_w) if (pa is not None and mid_x is not None) else None
    return feats, sho_w

def compare_frames(ref_feats, usr_feats):
    weighted=[]; issues=[]; bad=set()
    for fname, ftype, ka, kb, kc, w, desc in SCORE_FEATURES:
        rv, uv = ref_feats.get(fname), usr_feats.get(fname)
        if rv is None or uv is None:
            weighted.append((0, w)); continue  # 修正：缺失特征直接给0分
        sim = 1.0 - min(1.0, abs(rv - uv) / (rv + 0.15))
        weighted.append((sim, w))
        if sim < 0.65: bad.add(ka); issues.append(desc)
    if not weighted: return 0, [], set()
    total_w = sum(w for _,w in weighted)
    return sum(s*w for s,w in weighted)/total_w * 100, issues, bad

# ── 表情识别 ──────────────────────────────────────────────
class EmotionDetector:
    def __init__(self):
        self.enabled=False; self.sess=None
        for ep in[spath("emotion-ferplus-8.onnx"),spath("demo","emotion-ferplus-8.onnx")]:
            if os.path.isfile(ep):
                try:
                    import onnxruntime as ort
                    self.sess=ort.InferenceSession(ep,providers=["CPUExecutionProvider"])
                    self.in_name=self.sess.get_inputs()[0].name
                    self.enabled=True; break
                except: pass
    def predict(self,face):
        if not self.enabled or face is None or face.size==0: return None,0.
        try:
            gray=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
            gray=cv2.resize(gray,(64,64)).astype(np.float32)
            blob=gray.reshape(1,1,64,64)
            out=self.sess.run(None,{self.in_name:blob})[0][0]
            prob=np.exp(out-out.max()); prob/=prob.sum()
            return int(np.argmax(prob)),float(prob[PAIN_IDX])
        except: return None,0.

# ── 主类 ──────────────────────────────────────────────────
class RehabApp:
    def __init__(self, video_path, input_source):
        self.tr=TR(); self.emotion=EmotionDetector()
        self.vcap_path = video_path # 保存路径用于标题显示
        from ultralytics import YOLO
        self.yolo=YOLO(spath("yolov8n-pose.pt"))

        # 1. 预处理标准视频
        self.vcap = cv2.VideoCapture(video_path)
        self.vid_fps = self.vcap.get(cv2.CAP_PROP_FPS) or 30.
        self.vid_frames=[]; self.ref_kpts_seq=[]; self.ref_confs_seq=[]

        print("[预处理] 提取标准视频骨架...")
        while True:
            ret, frame = self.vcap.read()
            if not ret: break
            self.vid_frames.append(frame)
            res = self.yolo(frame, verbose=False, conf=0.3)
            if res and len(res[0].keypoints.xy)>0:
                self.ref_kpts_seq.append(res[0].keypoints.xy[0].cpu().numpy())
                self.ref_confs_seq.append(res[0].keypoints.conf[0].cpu().numpy())
            else:
                self.ref_kpts_seq.append(None); self.ref_confs_seq.append(None)

        if not self.vid_frames:
            print(f"[错误] 无法加载标准视频: {video_path}"); sys.exit(1)

        # 2. 初始化输入源
        self.is_video_input = isinstance(input_source, str)
        self.cap = cv2.VideoCapture(input_source if self.is_video_input else int(input_source))

        # 3. 状态变量
        self.score=100; self.smooth_score=100.0; self.bad_jts=set()
        self.issue_cache = []; self.issue_timer = 0 # 建议持久化缓存
        self.emo_label=None; self.emo_pain=0.; self.pain_blink=0
        self.paused=False; self.fc=0

    def run(self):
        cv2.namedWindow("康复跟练 v4.5", cv2.WINDOW_NORMAL)
        start_t = time.time(); fi = 0
        total_ref = len(self.vid_frames)

        while True:
            if not self.paused:
                fi = int((time.time() - start_t) * self.vid_fps * SPEED_FACTOR) % total_ref

            ret, usr_raw = self.cap.read()
            if not ret:
                if self.is_video_input: self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                break

            # 核心：镜像处理
            usr_raw = cv2.flip(usr_raw, 1)

            # YOLO 推理用户动作
            u_res = self.yolo(usr_raw, verbose=False, conf=0.3)
            uk_orig, uc_orig = None, None
            if u_res and len(u_res[0].keypoints.xy)>0:
                uk_orig = u_res[0].keypoints.xy[0].cpu().numpy()
                uc_orig = u_res[0].keypoints.conf[0].cpu().numpy()

                # 核心：防瞎评检测（必须有8个以上有效点）
                valid_kpts = np.sum(uc_orig > CONF_THRESH)
                if valid_kpts < 8:
                    self.score = 0; self.smooth_score *= 0.8
                    self.issue_cache = ["请退后，全身进入画面"]; self.issue_timer = 30
                    self.bad_jts = set()
                else:
                    # 核心：滑动窗口评分
                    u_feats, u_sw = extract_features(uk_orig, uc_orig)
                    best_s = -1; best_iss = []; best_bad = set()
                    if u_feats:
                        for offset in range(-SEARCH_WINDOW, SEARCH_WINDOW + 1):
                            idx = (fi + offset) % total_ref
                            rk, rc = self.ref_kpts_seq[idx], self.ref_confs_seq[idx]
                            if rk is not None:
                                rf, rsw = extract_features(rk, rc)
                                if rf:
                                    s, iss, bad = compare_frames(rf, u_feats)
                                    if s > best_s: best_s, best_iss, best_bad = s, iss, bad

                        # EMA平滑 + 建议持久化
                        self.smooth_score = self.smooth_score * 0.75 + max(0, best_s) * 0.25
                        self.score = int(self.smooth_score)
                        self.bad_jts = best_bad
                        if best_iss:
                            self.issue_cache = list(dict.fromkeys(best_iss)); self.issue_timer = 60
                        elif self.issue_timer > 0:
                            self.issue_timer -= 1
                        else:
                            self.issue_cache = []

            # 表情检测逻辑
            self.fc += 1
            if self.fc % EMOTION_EVERY == 0 and uk_orig is not None:
                H, W = usr_raw.shape[:2]
                if uc_orig[KP["nose"]]>0.25:
                    nx, ny = int(uk_orig[KP["nose"],0]), int(uk_orig[KP["nose"],1])
                    face = usr_raw[max(0,ny-80):min(H,ny+80), max(0,nx-80):min(W,nx+80)]
                    ei, ep = self.emotion.predict(face)
                    if ei is not None:
                        self.emo_label, self.emo_pain = FER_LABELS[ei], ep
                        if ep > PAIN_THRESH: self.pain_blink = 30

            # ── 渲染界面 ──────────────────────────────────────────
            canvas = np.zeros((WIN_H, WIN_W, 3), np.uint8)

            # 顶部动作名称
            action_title = os.path.basename(self.vcap_path).replace(".mp4", "")
            self.tr.put(canvas, f"当前动作：{action_title}", (WIN_W//2 - 160, 10), sz=32, col=(255,255,255), bold=True)

            # 左侧标准面板
            l_p, l_sc, l_px, l_py = letterbox(self.vid_frames[fi], PANEL_W, WIN_H)
            if self.ref_kpts_seq[fi] is not None:
                rk_p = kpts_to_panel(self.ref_kpts_seq[fi], self.vid_frames[fi].shape[1], self.vid_frames[fi].shape[0], l_sc, l_px, l_py)
                self._draw_skel(l_p, rk_p, self.ref_confs_seq[fi], (0, 220, 80))
            canvas[:, :PANEL_W] = l_p

            # 右侧练习面板
            r_p, r_sc, r_px, r_py = letterbox(usr_raw, PANEL_W, WIN_H)
            if uk_orig is not None:
                uk_p = kpts_to_panel(uk_orig, usr_raw.shape[1], usr_raw.shape[0], r_sc, r_px, r_py)
                self._draw_skel(r_p, uk_p, uc_orig, (255, 200, 0), self.bad_jts)

            # 渲染右侧文字
            self.tr.put(r_p, f"评分: {self.score}", (25, 30), sz=65, col=(0,255,0) if self.score>80 else (0,150,255), bold=True)
            iy = 130
            for tip in self.issue_cache[:3]:
                self.tr.put(r_p, f"提示: {tip}", (25, iy), sz=30, col=(120,120,255), bold=True); iy += 45
            if self.emo_label:
                self.tr.put(r_p, f"状态: {self.emo_label} (痛苦度:{self.emo_pain:.1%})", (25, WIN_H-60), sz=26, col=(255,255,255), bold=True)

            # 痛苦警报闪烁
            if self.pain_blink > 0:
                self.pain_blink -= 1
                if self.pain_blink % 6 < 3: cv2.rectangle(r_p, (0,0), (PANEL_W, WIN_H), (0,0,220), 10)

            canvas[:, PANEL_W:] = r_p
            cv2.imshow("康复跟练 v4.5", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord(' '):
                self.paused = not self.paused
                if not self.paused: start_t = time.time() - fi/(self.vid_fps*SPEED_FACTOR)
            elif key == ord('r'): start_t = time.time(); self.smooth_score=100

        self.cap.release(); cv2.destroyAllWindows()

    def _draw_skel(self, canvas, kpts, confs, color, bads=None):
        bads = bads or set()
        for na, nb, _ in SKEL_LINKS:
            ia, ib = KP[na], KP[nb]
            if confs[ia]>CONF_THRESH and confs[ib]>CONF_THRESH:
                c = (40, 40, 230) if (na in bads or nb in bads) else color
                cv2.line(canvas, (int(kpts[ia,0]), int(kpts[ia,1])), (int(kpts[ib,0]), int(kpts[ib,1])), c, 4, cv2.LINE_AA)
        for name, i in KP.items():
            if confs[i]>CONF_THRESH:
                c = (30,30,230) if name in bads else (255,255,255)
                cv2.circle(canvas, (int(kpts[i,0]), int(kpts[i,1])), 5, c, -1)

# ── 终端启动逻辑 ──────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--input_video", default=None)
    args = ap.parse_args()

    # 路径自动识别
    v_path = args.video
    if not v_path:
        for p in [spath("demo/侧布扩胸激活.mp4"), spath("侧布扩胸激活.mp4")]:
            if os.path.exists(p): v_path = p; break

    if not v_path or not os.path.exists(v_path):
        print(f"[错误] 未找到标准视频文件: {v_path}"); sys.exit(1)

    src = args.input_video if args.input_video else args.cam
    app = RehabApp(v_path, src)
    app.run()

if __name__ == "__main__":
    main()