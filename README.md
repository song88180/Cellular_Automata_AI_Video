# 元胞自动机 ASCII 科普短片 / Cellular Automata, in your terminal

一个在终端里播放的元胞自动机科普短片：纯字符动画（ANSI truecolor + 半块字符方形像素）、中英双语字幕、
程序合成的配乐。风格参考 [world.execute-me-ascii](https://github.com/yym8224961/world.execute-me-ascii)。
每一章都是**真实计算**的模拟，不是预录动画——包括第 6 章里真正训练出来的神经元胞自动机。

全片约 10 分 11 秒（1920×1080 / 24 fps）。字幕和画面上的图解都写得足够完整——暂停下来就能看懂每一条规则。

| 章节 | 内容 |
|---|---|
| 1 起源（0′55″） | 像素标题按生命游戏规则演化，镜头一路推近直到方块全部飞出画面；「01 起源」；冯·诺依曼人物卡；**真实运行的冯·诺依曼 29 态元胞自动机**：信号环 + 汇合态 + 构造臂，指令 10000（建右箭头）与 1010（建下箭头）让构造臂一边建造、一边转弯；元胞自动机的定义，引出一维 |
| 2 一维元胞自动机（2′09″） | 一排格子、三格邻域、下一代先是一排「?」；Rule 22 的 8 条规则 + 繁衍/拥挤/死亡比喻、规则编号的二进制由来；滑动窗口逐格演算 → 加速 → 缩放出谢尔宾斯基三角形；Wolfram 人物卡；Rule 30（缩放到 1 格 = 1 像素后镜头向下跟随；海螺花纹对照，以字符画呈现）；Rule 110（始终最高分辨率自上而下演化）；Wolfram 四类行为 |
| 3 生命游戏（2′03″） | Conway 人物卡；3×3 摩尔邻域（大号数字）；四条规则逐一演示（B3/S23）；大网格逐格计数再同步更新；静物与振荡器；滑翔机四相位 + 两架滑翔机与一艘轻型飞船互不干扰地飞行；高斯帕滑翔机枪；随机初始状态 |
| 4 Langton 蚂蚁与 turmite（1′17″） | Langton 人物卡；规则卡逐步演示 → 一万步后的「高速公路」；RLLR 对称生长；7 状态龙形曲线（规则取自 Greg Turk） |
| 5 自复制环（1′22″） | 冯·诺依曼 → Codd → Langton 86 格的环；151 步复制出子环，停在这一帧；群落像珊瑚一样生长；**Evoloop**（Sayama 1999）：第 9 种「溶解」态，环相撞、变异、被选择，出现进化 |
| 6 Growing Neural CA（1′29″） | 原理逐条图解（网络只画出可读的一部分，连线是真实权重）；从种子开始生长的全过程（镜头固定、种子在底部，每一步是否更新都画出来）；隐藏通道同时发育；擦除与切半后自我修复 |
| 7 总结（0′56″） | 五个系统同屏回顾（Rule 30、生命游戏、经典兰顿蚂蚁、单个自复制环、会被咬掉又长回来的蜥蜴）；参考文献与致谢；作者卡；「感谢观看」按生命游戏演化——**终端里会一直演化，按任意键退出**，视频则淡出结束 |

**人物肖像**：人物已抠图去背景（`tools/make_portrait_masks.py`，rembg / BiRefNet-portrait，仅开发时需要，结果存为
`assets/images/*_mask.png`），再用 Floyd–Steinberg 抖动映射成盲文点阵字符（每个字符 2×4 个点），40 行高的肖像
保留 160 行的细节。人名和章节中文标题用「元胞自动机逆向演化」显现：先把文字按生命游戏正向演化若干代，再倒着播放
回文字，同时淡入。想换回原来的色块肖像：把 `chapters/common.py` 里的 `PORTRAIT_STYLE` 改成 `"pixels"`。

## 在终端里看

```bash
pip install numpy pillow
python3 main.py                        # 全片
python3 main.py --chapter chapter4     # 单独一章（可逗号分隔多章）
python3 main.py --at 90                # 从第 90 秒开始
python3 main.py --mute                 # 不要声音
```

| 按键 | 作用 |
|---|---|
| `space` | 暂停 / 继续 |
| `←` / `→` | 后退 / 前进 5 秒 |
| `esc`（或 `q`） | 退出 |
| 片尾「感谢观看」 | 会一直演化，按任意键退出 |

- 画面固定为 160×40 字符（+ 字幕和状态栏共 44 行），**自动居中**，整屏铺满影片背景色。终端小于 160×44 时会提示放大窗口或缩小字体（`Ctrl -`）。
- 第一次运行会先把所有模拟计算一遍（各章并行，约一两分钟），结果缓存在 `build/cache/`；之后秒开，前进后退也不需要重新计算。改了哪一章，只重算那一章。
- 配乐通过 `aplay` / `pw-play` / `paplay` 播放（有哪个用哪个）；都没有时静音播放，状态栏会提示。
- 终端需支持 24-bit 真彩色（绝大多数现代终端都支持）。

## 导出视频（1920×1080 mp4，含配乐）

```bash
python3 main.py --render build/out/final.mp4 --srt build/out/final.srt
```

- 需要 `ffmpeg`。程序直接把每一帧光栅化（12×24 像素的字符格，方块字符按像素精确绘制，半块像素是正方形），
  用 ffmpeg 编码，并混入配乐。
- 160×40 的字符画面铺满 16:9 画面上方 1920×960，底部 120 像素是**大号中英字幕**（Noto Sans CJK，中文 40px，
  英文 27px，切换时淡入淡出）。明亮的格子带一点荧光辉光（`--no-glow` 可关）。
- `--srt` 另外导出外挂字幕。
- 预览用：`--sheet sheet.png --every 2 [--start S --end E]` 输出缩略图联系表，`--still 12.5 --out f.png` 输出单帧。

## 配乐

`engine/audio.py` 用 numpy 实时合成（没有使用任何外部音频素材）：

- 每个场景声明一种情绪（`title / calm / curious / build / chaos / wonder / epic / tense / outro`），决定和弦进行、
  pad / 琶音 / 贝斯 / 钟声 / 鼓的配比和密度；情绪之间按小节平滑过渡。
- 画面里发生的事件会在**同一帧**触发音效：细胞诞生（音高吸附到当前和弦上）、逐步演示的滴答声、标题锁定的重击、
  擦除时的刷声、转场的呼啸声、高潮前的上升音等。
- 结果缓存为 `build/cache/*.wav`，终端播放和导出视频共用同一条音轨。

## 重新训练 Growing NCA（可选，仅开发时需要）

仓库里已经带了训练好的权重 `sim/nca/weights/model.npz`，播放时只用 numpy 推理。想重新训练：

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m sim.nca.train --steps 8000          # GPU 约 15 分钟
```

## 测试

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/
```

包括：Rule 90 与 Lucas 定理逐格比对、各种静物/振荡器的周期、滑翔机与 LWSS 的平移、滑翔机枪每 30 代 +5 个细胞、
Langton 蚂蚁 104 步周期的「高速公路」、Langton 环 151 步后出现完整子环、numpy 与 torch 版 NCA 数值一致（1e-5）、
训练好的 NCA 能生长且能再生，以及引擎本身（帧打包往返、转场、拼接、光栅化像素精确、配乐无溢出）。
（`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 是为了绕开本机 conda 环境里一个无关的坏 pytest 插件；torch 相关测试需用 `.venv/bin/python`。）

## 项目结构

```
main.py        CLI：播放 / 导出 / 预览
engine/        frame.py（帧与场景）  canvas.py（方形像素画布、字体、配色、缓动）  fx.py（转场/揭示特效）
               timeline.py（全片时间线、逐章并行计算与缓存、拼接）  player.py（终端播放器：居中、暂停、快进快退、声音）
               raster.py（帧 → 1920×1080 图像、大字幕、辉光）  export.py（mp4 / 联系表 / 单帧）
               audio.py（程序化配乐与音效）  ascii_art.py（肖像：ASCII 字符“解码”成像素照片）
sim/           纯计算：elementary_ca / game_of_life / turmite / von_neumann / langton_loops / evoloop /
               nca（model.py 推理, train.py 训练）
chapters/      每章一个模块，build_scenes(rows, cols) -> List[Scene]；common.py 为章节标题卡与人物卡
assets/        人物与海螺照片（仅以字符/像素画形式出现，见 assets/images/CREDITS.txt）、Langton 环规则表
tests/         单元测试
```

## 参考与致谢

- J. von Neumann, A. W. Burks (ed.), *Theory of Self-Reproducing Automata* (1966)；E. F. Codd, *Cellular Automata* (1968)
- S. Wolfram, "Statistical mechanics of cellular automata", *Rev. Mod. Phys.* 55 (1983)；*A New Kind of Science* (2002)；M. Cook, "Universality in Elementary Cellular Automata", *Complex Systems* 15 (2004)
- M. Gardner, "Mathematical Games", *Scientific American* (Oct 1970)；R. W. Gosper 滑翔机枪 (1970)；A. Wade, Gemini (2010)
- C. G. Langton, "Self-reproduction in cellular automata", *Physica D* 10 (1984)；"Studying artificial life with cellular automata", *Physica D* 22 (1986)
- G. Turk, *Vants and Turmites* — https://github.com/gturk1/turmites （第 4 章的规则表均取自该仓库）
- Golly 的 Langtons-Loops 规则表与初始图案（`assets/rules/`）
- A. Mordvintsev, E. Randazzo, E. Niklasson, M. Levin, "Growing Neural Cellular Automata", *Distill* (2020) — https://distill.pub/2020/growing-ca/
- 人物与海螺照片：Wikimedia Commons（Conway：Denise Applewhite, Office of Communications；LANL；S. Faust CC BY-SA 3.0；Cary Herz CC BY-SA 3.0；Richard Ling CC BY-SA 3.0），详见 `assets/images/CREDITS.txt`
- 蜥蜴目标图：Google Noto Emoji（Apache-2.0）；配乐与音效：numpy 程序合成

## 许可

代码使用 [GPLv3](LICENSE) 许可；`assets/` 下的第三方素材许可见 `assets/images/CREDITS.txt` 与
`assets/rules/README.txt`。
