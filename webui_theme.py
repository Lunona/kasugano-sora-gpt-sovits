"""Kasugano Sora (Yosuga no Sora) Gradio theme, CSS, SVG motifs, and static UI asset URLs."""
from pathlib import Path
from urllib.parse import quote

import gradio as gr

UI_ASSETS_DIR = Path(__file__).resolve().parent / "ui_assets"

UI_THEME = gr.themes.Soft(
    primary_hue="sky",
    secondary_hue="slate",
    neutral_hue="slate",
    radius_size="lg",
    font=["Microsoft YaHei UI", "Hiragino Sans", "PingFang SC", "Noto Sans SC", "sans-serif"],
    font_mono=["Cascadia Code", "Consolas", "monospace"],
).set(
    body_background_fill="linear-gradient(155deg, #f2f7fd 0%, #f8fafd 52%, #e9f2fb 100%)",
    body_background_fill_dark="linear-gradient(155deg, #0b1020 0%, #11182c 55%, #161e34 100%)",
    block_background_fill="rgba(255, 255, 255, 0.86)",
    block_background_fill_dark="rgba(17, 23, 39, 0.82)",
    block_border_width="1px",
    block_border_color="rgba(148, 179, 214, 0.42)",
    block_border_color_dark="rgba(148, 179, 214, 0.22)",
    block_label_background_fill="rgba(242, 247, 253, 0.94)",
    block_label_background_fill_dark="rgba(24, 34, 58, 0.92)",
    block_label_text_color="#25324a",
    block_label_text_color_dark="#d6e4f7",
    block_label_border_width="1px",
    block_label_border_color="rgba(148, 179, 214, 0.45)",
    block_label_border_color_dark="rgba(138, 184, 232, 0.28)",
    block_title_background_fill="rgba(242, 247, 253, 0.94)",
    block_title_background_fill_dark="rgba(24, 34, 58, 0.92)",
    block_title_text_color="#25324a",
    block_title_text_color_dark="#d6e4f7",
    border_color_primary="rgba(148, 179, 214, 0.35)",
    border_color_primary_dark="rgba(30, 42, 68, 0.72)",
    background_fill_secondary="rgba(244, 249, 255, 0.86)",
    background_fill_secondary_dark="rgba(16, 22, 38, 0.86)",
    block_shadow="0 6px 24px rgba(58, 92, 138, 0.08)",
    input_background_fill="rgba(255, 255, 255, 0.94)",
    input_background_fill_dark="rgba(13, 19, 34, 0.88)",
    input_border_color="rgba(148, 179, 214, 0.48)",
    input_border_color_dark="rgba(148, 179, 214, 0.26)",
    input_border_color_focus="#5b8ec9",
    button_large_text_weight="800",
    button_primary_background_fill="linear-gradient(110deg, #252d42 0%, #4676ae 58%, #6ea4dc 100%)",
    button_primary_background_fill_hover="linear-gradient(110deg, #1d2436 0%, #3a689e 58%, #5d95d0 100%)",
    button_primary_text_color="#ffffff",
    button_primary_border_color="transparent",
    slider_color="#4b7cb5",
    slider_color_dark="#7daee0",
    checkbox_background_color_selected="#2b3854",
    checkbox_border_color_selected="#2b3854",
    checkbox_label_background_fill_selected="linear-gradient(108deg, #232a3d 0%, #4472a8 70%, #6396ce 100%)",
    checkbox_label_background_fill_selected_dark="linear-gradient(108deg, #232a3d 0%, #4472a8 70%, #6396ce 100%)",
    checkbox_label_text_color_selected="#ffffff",
    checkbox_label_text_color_selected_dark="#ffffff",
)

def _ui_asset_url(filename: str) -> str:
    """Return Gradio static /file= URL for a file under ui_assets/ (cacheable, avoids MB-scale inline CSS)."""
    path = UI_ASSETS_DIR / filename
    if not path.is_file():
        return ""
    return f"/file={quote(path.as_posix(), safe='/:')}"


_BG_LIGHT_URI = _ui_asset_url("sora_bg_light.jpg")
_BG_DARK_URI = _ui_asset_url("sora_bg_dark.jpg")
_HERO_LIGHT_URI = (
    _ui_asset_url("sora_hero_light_comfy.jpg")
    or _ui_asset_url("sora_hero_light.jpg")
    or _ui_asset_url("sora_hero_bg.jpg")
)
_HERO_DARK_URI = (
    _ui_asset_url("sora_hero_dark_comfy.jpg")
    or _ui_asset_url("sora_hero_dark.jpg")
    or _HERO_LIGHT_URI
)
_AVATAR_URI = (
    _ui_asset_url("sora_avatar_comfy.jpg")
    or _ui_asset_url("sora_avatar_v2.jpg")
    or _ui_asset_url("sora_avatar.jpg")
)
_PRESET_DAILY_URI = _ui_asset_url("sora_preset_daily.jpg")
_PRESET_CLINGY_URI = _ui_asset_url("sora_preset_clingy.jpg")
_PRESET_TSUN_URI = _ui_asset_url("sora_preset_tsun.jpg")
_PRESET_ASMR_URI = _ui_asset_url("sora_preset_asmr.jpg")
_GEN_BTN_ART_URI = _ui_asset_url("sora_gen_btn_art.jpg")
_PLAYER_BG_URI = _ui_asset_url("sora_player_bg.jpg")
_CHIBI_STICKER_URI = _ui_asset_url("sora_chibi_sticker.png")
_CHIBI_WRITE_URI = _ui_asset_url("sora_chibi_write.png")
_FB_GOOD_URI = _ui_asset_url("sora_fb_good.jpg")
_FB_BAD_URI = _ui_asset_url("sora_fb_bad.jpg")
_TAB_SELFOPT_URI = _ui_asset_url("sora_tab_selfopt.jpg")
_TAB_API_URI = _ui_asset_url("sora_tab_api.jpg")
_TAB_FX_URI = _ui_asset_url("sora_tab_fx.jpg")
_TAB_ADV_URI = _ui_asset_url("sora_tab_adv.jpg")
_CHAT_BG_URI = _ui_asset_url("sora_chat_bg.jpg")

_BG_WALLPAPER_CSS = ""
if _BG_LIGHT_URI:
    _BG_WALLPAPER_CSS += f"""
body, .gradio-container {{
  background-image:
    linear-gradient(155deg, rgba(244, 248, 253, 0.80) 0%, rgba(250, 251, 255, 0.76) 52%, rgba(236, 244, 253, 0.82) 100%),
    url("{_BG_LIGHT_URI}") !important;
  background-size: cover !important;
  background-position: center top !important;
  background-attachment: fixed !important;
  background-repeat: no-repeat !important;
}}
"""
if _BG_DARK_URI:
    _BG_WALLPAPER_CSS += f"""
.dark body, .dark .gradio-container {{
  background-image:
    linear-gradient(155deg, rgba(10, 15, 28, 0.78) 0%, rgba(15, 21, 38, 0.74) 55%, rgba(19, 25, 44, 0.82) 100%),
    url("{_BG_DARK_URI}") !important;
  background-size: cover !important;
  background-position: center top !important;
  background-attachment: fixed !important;
  background-repeat: no-repeat !important;
}}
"""
if _HERO_LIGHT_URI:
    _BG_WALLPAPER_CSS += f"""
#sora-hero {{
  min-height: 158px;
  padding-right: 300px;
  background-image:
    linear-gradient(92deg, rgba(252, 253, 255, 0.92) 0%, rgba(246, 251, 255, 0.80) 44%, rgba(240, 248, 255, 0.24) 68%, rgba(255, 255, 255, 0.0) 86%),
    url("{_HERO_LIGHT_URI}") !important;
  background-size: 100% 100%, cover !important;
  background-position: center, center 22% !important;
  background-repeat: no-repeat, no-repeat !important;
}}
"""
if _HERO_DARK_URI:
    _BG_WALLPAPER_CSS += f"""
.dark #sora-hero {{
  background-image:
    linear-gradient(92deg, rgba(11, 16, 30, 0.90) 0%, rgba(15, 22, 40, 0.74) 46%, rgba(16, 24, 44, 0.18) 70%, rgba(14, 19, 34, 0.0) 88%),
    url("{_HERO_DARK_URI}") !important;
  background-size: 100% 100%, cover !important;
  background-position: center, center 24% !important;
  background-repeat: no-repeat, no-repeat !important;
}}
"""
_BG_WALLPAPER_CSS += """
@media (max-width: 960px) {
  #sora-hero { padding-right: 24px !important; }
}
"""
if _GEN_BTN_ART_URI:
    _BG_WALLPAPER_CSS += f"""
#sora-gen-btn,
.gradio-container button.sora-art-primary {{
  background-image:
    linear-gradient(90deg, rgba(20, 32, 58, 0.44) 0%, rgba(22, 42, 78, 0.54) 50%, rgba(20, 32, 58, 0.40) 100%),
    url("{_GEN_BTN_ART_URI}") !important;
  background-size: 100% 100%, cover !important;
  background-position: center, center 42% !important;
  background-repeat: no-repeat, no-repeat !important;
  text-shadow: 0 2px 10px rgba(10, 20, 44, 0.72) !important;
}}
"""
if _CHAT_BG_URI:
    _BG_WALLPAPER_CSS += f"""
.sora-chat {{
  background-image:
    linear-gradient(180deg, rgba(250, 253, 255, 0.86) 0%, rgba(244, 249, 255, 0.84) 100%),
    url("{_CHAT_BG_URI}") !important;
  background-size: 100% 100%, cover !important;
  background-position: center, center !important;
  background-repeat: no-repeat, no-repeat !important;
}}
.dark .sora-chat {{
  background-image:
    linear-gradient(180deg, rgba(13, 18, 32, 0.88) 0%, rgba(16, 23, 40, 0.86) 100%),
    url("{_CHAT_BG_URI}") !important;
  background-size: 100% 100%, cover !important;
  background-position: center, center !important;
  background-repeat: no-repeat, no-repeat !important;
}}
"""
if _PLAYER_BG_URI:
    _BG_WALLPAPER_CSS += f"""
.sora-audio-box {{
  background-image:
    linear-gradient(180deg, rgba(255, 255, 255, 0.72) 0%, rgba(242, 248, 255, 0.66) 100%),
    url("{_PLAYER_BG_URI}") !important;
  background-size: 100% 100%, cover !important;
  background-position: center, center 62% !important;
  background-repeat: no-repeat, no-repeat !important;
}}
.dark .sora-audio-box {{
  background-image:
    linear-gradient(180deg, rgba(16, 22, 38, 0.80) 0%, rgba(12, 18, 32, 0.76) 100%),
    url("{_PLAYER_BG_URI}") !important;
  background-size: 100% 100%, cover !important;
  background-position: center, center 62% !important;
  background-repeat: no-repeat, no-repeat !important;
}}
"""
if _PRESET_DAILY_URI:
    _BG_WALLPAPER_CSS += f"""
.gradio-container button.sora-preset-daily {{
  background-image:
    linear-gradient(90deg, rgba(255, 255, 255, 0.98) 0%, rgba(240, 248, 255, 0.96) calc(100% - 148px), rgba(240, 248, 255, 0.72) calc(100% - 110px), transparent calc(100% - 58px)),
    url("{_PRESET_DAILY_URI}") !important;
  background-size: 100% 100%, auto 112% !important;
  background-position: center, right -4px top 8% !important;
  background-repeat: no-repeat, no-repeat !important;
}}
.dark .gradio-container button.sora-preset-daily {{
  background-image:
    linear-gradient(90deg, rgba(18, 28, 48, 0.98) 0%, rgba(22, 36, 62, 0.96) calc(100% - 148px), rgba(22, 36, 62, 0.70) calc(100% - 110px), rgba(18, 28, 48, 0.06) calc(100% - 54px)),
    url("{_PRESET_DAILY_URI}") !important;
  background-size: 100% 100%, auto 112% !important;
  background-position: center, right -4px top 8% !important;
  background-repeat: no-repeat, no-repeat !important;
}}
"""
if _PRESET_CLINGY_URI:
    _BG_WALLPAPER_CSS += f"""
.gradio-container button.sora-preset-clingy {{
  background-image:
    linear-gradient(90deg, rgba(255, 250, 252, 0.98) 0%, rgba(254, 242, 247, 0.96) calc(100% - 148px), rgba(254, 242, 247, 0.72) calc(100% - 110px), transparent calc(100% - 58px)),
    url("{_PRESET_CLINGY_URI}") !important;
  background-size: 100% 100%, auto 114% !important;
  background-position: center, right -4px top 6% !important;
  background-repeat: no-repeat, no-repeat !important;
}}
.dark .gradio-container button.sora-preset-clingy {{
  background-image:
    linear-gradient(90deg, rgba(44, 24, 38, 0.98) 0%, rgba(52, 28, 44, 0.96) calc(100% - 148px), rgba(52, 28, 44, 0.70) calc(100% - 110px), rgba(44, 24, 38, 0.06) calc(100% - 54px)),
    url("{_PRESET_CLINGY_URI}") !important;
  background-size: 100% 100%, auto 114% !important;
  background-position: center, right -4px top 6% !important;
  background-repeat: no-repeat, no-repeat !important;
}}
"""
if _PRESET_TSUN_URI:
    _BG_WALLPAPER_CSS += f"""
.gradio-container button.sora-preset-tsun {{
  background-image:
    linear-gradient(90deg, rgba(248, 250, 253, 0.98) 0%, rgba(240, 243, 248, 0.96) calc(100% - 148px), rgba(240, 243, 248, 0.72) calc(100% - 110px), transparent calc(100% - 58px)),
    url("{_PRESET_TSUN_URI}") !important;
  background-size: 100% 100%, auto 112% !important;
  background-position: center, right -12px top 10% !important;
  background-repeat: no-repeat, no-repeat !important;
}}
.dark .gradio-container button.sora-preset-tsun {{
  background-image:
    linear-gradient(90deg, rgba(24, 28, 42, 0.98) 0%, rgba(28, 34, 50, 0.96) calc(100% - 148px), rgba(28, 34, 50, 0.70) calc(100% - 110px), rgba(24, 28, 42, 0.06) calc(100% - 54px)),
    url("{_PRESET_TSUN_URI}") !important;
  background-size: 100% 100%, auto 112% !important;
  background-position: center, right -12px top 10% !important;
  background-repeat: no-repeat, no-repeat !important;
}}
"""
if _PRESET_ASMR_URI:
    _BG_WALLPAPER_CSS += f"""
.gradio-container button.sora-preset-asmr {{
  background-image:
    linear-gradient(90deg, rgba(16, 24, 46, 0.99) 0%, rgba(20, 32, 62, 0.98) calc(100% - 148px), rgba(20, 32, 62, 0.72) calc(100% - 108px), transparent calc(100% - 54px)),
    url("{_PRESET_ASMR_URI}") !important;
  background-size: 100% 100%, auto 114% !important;
  background-position: center, right -4px top 8% !important;
  background-repeat: no-repeat, no-repeat !important;
  text-shadow: 0 1px 6px rgba(8, 14, 30, 0.85) !important;
}}
"""
if _FB_GOOD_URI:
    _BG_WALLPAPER_CSS += f"""
.gradio-container button.sora-fb-good {{
  min-height: 48px !important;
  justify-content: flex-start !important;
  padding: 6px 14px !important;
  background-image:
    linear-gradient(92deg, rgba(255, 255, 255, 0.96) 0%, rgba(240, 248, 255, 0.90) 52%, rgba(240, 248, 255, 0.16) 78%, transparent 100%),
    url("{_FB_GOOD_URI}") !important;
  background-size: 100% 100%, auto 115% !important;
  background-position: center, right 4px top 12% !important;
  background-repeat: no-repeat, no-repeat !important;
}}
.dark .gradio-container button.sora-fb-good {{
  background-image:
    linear-gradient(92deg, rgba(20, 30, 52, 0.96) 0%, rgba(24, 38, 66, 0.88) 54%, rgba(24, 38, 66, 0.22) 80%, rgba(20, 30, 52, 0.08) 100%),
    url("{_FB_GOOD_URI}") !important;
  background-size: 100% 100%, auto 115% !important;
  background-position: center, right 4px top 12% !important;
  background-repeat: no-repeat, no-repeat !important;
}}
"""
if _FB_BAD_URI:
    _BG_WALLPAPER_CSS += f"""
.gradio-container button.sora-fb-bad {{
  min-height: 48px !important;
  justify-content: flex-start !important;
  padding: 6px 14px !important;
  background-image:
    linear-gradient(92deg, rgba(255, 247, 249, 0.96) 0%, rgba(253, 238, 243, 0.90) 52%, rgba(253, 238, 243, 0.14) 78%, transparent 100%),
    url("{_FB_BAD_URI}") !important;
  background-size: 100% 100%, auto 115% !important;
  background-position: center, right 2px top 8% !important;
  background-repeat: no-repeat, no-repeat !important;
}}
.dark .gradio-container button.sora-fb-bad {{
  background-image:
    linear-gradient(92deg, rgba(56, 22, 36, 0.96) 0%, rgba(62, 26, 40, 0.88) 54%, rgba(62, 26, 40, 0.22) 80%, rgba(56, 22, 36, 0.08) 100%),
    url("{_FB_BAD_URI}") !important;
  background-size: 100% 100%, auto 115% !important;
  background-position: center, right 2px top 8% !important;
  background-repeat: no-repeat, no-repeat !important;
}}
"""
for _cls, _uri in (
    ("sora-banner-fx", _TAB_FX_URI),
    ("sora-banner-adv", _TAB_ADV_URI),
    ("sora-banner-selfopt", _TAB_SELFOPT_URI),
    ("sora-banner-api", _TAB_API_URI),
):
    if _uri:
        _BG_WALLPAPER_CSS += f"""
.{_cls} {{
  background-image:
    linear-gradient(90deg, rgba(252, 254, 255, 0.99) 0%, rgba(244, 249, 255, 0.98) calc(100% - 315px), rgba(244, 249, 255, 0.72) calc(100% - 235px), transparent calc(100% - 115px)),
    url("{_uri}") !important;
}}
.dark .{_cls} {{
  background-image:
    linear-gradient(90deg, rgba(16, 22, 38, 0.99) 0%, rgba(20, 28, 48, 0.98) calc(100% - 315px), rgba(20, 28, 48, 0.70) calc(100% - 235px), rgba(16, 22, 38, 0.04) calc(100% - 115px)),
    url("{_uri}") !important;
}}
"""

UI_CSS = """
/* ═══════════════════════════════════════════════════════════════
   春日野穹 · 缘之空 苍穹白裙与黑缎带视觉体系 (Kasugano Sora Theme)
   ═══════════════════════════════════════════════════════════════ */
:root {
  --sora-ink: #1e2536;
  --sora-ink-soft: #4a566e;
  --sora-ribbon: #232836;
  --sora-ribbon-hover: #31384c;
  --sora-sky: #5589c4;
  --sora-sky-deep: #346094;
  --sora-sky-soft: #e6f1fc;
  --sora-blush: #cf829a;
  --sora-border: rgba(148, 179, 214, 0.45);
  --sora-lace: rgba(255, 255, 255, 0.92);
  --sora-card-bg: rgba(255, 255, 255, 0.86);
  --sora-card-shadow: 0 8px 26px rgba(52, 86, 130, 0.09), inset 0 1px 0 rgba(255, 255, 255, 0.96);
  --sora-card-hover-shadow: 0 10px 30px rgba(52, 86, 130, 0.13), inset 0 1px 0 rgba(255, 255, 255, 1);
  --sora-subcard-bg: rgba(244, 249, 255, 0.72);
  --sora-serif: "Yu Mincho", "Hiragino Mincho ProN", "Songti SC", "STSong", "Noto Serif SC", Georgia, serif;
}
.dark {
  --sora-ink: #e8eef8;
  --sora-ink-soft: #a9b8d0;
  --sora-ribbon: #181e2e;
  --sora-ribbon-hover: #252e45;
  --sora-sky: #8ab8e8;
  --sora-sky-deep: #5d90c9;
  --sora-sky-soft: rgba(91, 142, 201, 0.16);
  --sora-blush: #e29db2;
  --sora-border: rgba(148, 179, 214, 0.24);
  --sora-lace: rgba(186, 210, 240, 0.16);
  --sora-card-bg: rgba(16, 22, 38, 0.84);
  --sora-card-shadow: 0 10px 32px rgba(0, 0, 0, 0.36), inset 0 1px 0 rgba(186, 210, 240, 0.10);
  --sora-card-hover-shadow: 0 12px 34px rgba(0, 0, 0, 0.42), inset 0 1px 0 rgba(186, 210, 240, 0.16);
  --sora-subcard-bg: rgba(22, 30, 50, 0.68);
}

.gradio-container {
  max-width: 1480px !important;
  margin: 0 auto !important;
  padding: 14px 20px 28px !important;
}
footer {display: none !important;}

/* 优雅自定义滚动条 */
::-webkit-scrollbar { width: 7px; height: 7px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
  background: rgba(125, 160, 200, 0.38);
  border-radius: 999px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(85, 137, 196, 0.62); }

/* ───────────── 顶部横幅（缘之空 · 晴空白裙 / 月夜微雨） ───────────── */
#sora-hero {
  position: relative;
  overflow: hidden;
  background-color: rgba(250, 252, 255, 0.92);
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
  border: 1px solid rgba(172, 198, 228, 0.65);
  border-radius: 24px;
  padding: 20px 28px;
  margin-bottom: 14px;
  box-shadow:
    0 12px 34px rgba(52, 86, 130, 0.12),
    inset 0 1px 0 rgba(255, 255, 255, 0.96),
    inset 0 0 0 4px rgba(255, 255, 255, 0.34);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}
.dark #sora-hero {
  background-color: rgba(14, 20, 36, 0.90);
  border-color: rgba(148, 179, 214, 0.28);
  box-shadow:
    0 14px 38px rgba(0, 0, 0, 0.44),
    inset 0 1px 0 rgba(196, 218, 245, 0.14),
    inset 0 0 0 4px rgba(148, 179, 214, 0.05);
}
/* 顶部纤细黑缎带装饰线 */
#sora-hero::before {
  content: "";
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: linear-gradient(90deg, #232836 0%, #4b7cb5 38%, #9dc4eb 72%, transparent 100%);
  opacity: 0.85;
  pointer-events: none;
}
.dark #sora-hero::before {
  background: linear-gradient(90deg, #8ab8e8 0%, #4b7cb5 45%, rgba(226, 157, 178, 0.55) 75%, transparent 100%);
  opacity: 0.72;
}
#sora-hero .hero-main {
  position: relative;
  z-index: 2;
  display: flex;
  align-items: center;
  gap: 18px;
  max-width: 100%;
}
#sora-hero .sora-crest-wrap {
  position: relative;
  flex-shrink: 0;
}
#sora-hero .sora-crest {
  width: 78px;
  height: 78px;
  border-radius: 50%;
  overflow: hidden;
  background: #ffffff;
  border: 3px solid #ffffff;
  box-shadow:
    0 6px 20px rgba(45, 76, 118, 0.24),
    0 0 0 1.5px rgba(35, 40, 54, 0.22);
  display: flex;
  align-items: center;
  justify-content: center;
}
#sora-hero .sora-crest img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.dark #sora-hero .sora-crest {
  border-color: rgba(203, 220, 242, 0.75);
  box-shadow:
    0 8px 24px rgba(0, 0, 0, 0.45),
    0 0 0 1.5px rgba(138, 184, 232, 0.35);
}
/* 徽章角落的小黑色蝴蝶结 */
#sora-hero .crest-bow {
  position: absolute;
  right: -4px;
  bottom: -2px;
  width: 28px;
  height: 22px;
  filter: drop-shadow(0 2px 4px rgba(15, 23, 42, 0.28));
  pointer-events: none;
}
#sora-hero .hero-copy {
  display: flex;
  flex-direction: column;
  gap: 3px;
}
#sora-hero .jp-kicker {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 2.4px;
  color: var(--sora-sky-deep);
  text-transform: uppercase;
}
.dark #sora-hero .jp-kicker { color: #9bc5ee; }
#sora-hero .title {
  font-family: var(--sora-serif);
  font-size: 27px;
  font-weight: 700;
  letter-spacing: 2px;
  color: var(--sora-ink);
  line-height: 1.2;
}
#sora-hero .title .title-sep {
  color: var(--sora-sky);
  font-weight: 400;
  margin: 0 6px;
}
#sora-hero .subtitle {
  font-size: 13px;
  color: var(--sora-ink-soft);
  letter-spacing: 0.3px;
  margin-top: 2px;
}
#sora-hero .badges {
  display: flex;
  gap: 7px;
  flex-wrap: wrap;
  margin-top: 8px;
}
#sora-hero .badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 11.5px;
  font-weight: 600;
  padding: 3.5px 11px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.86);
  border: 1px solid rgba(148, 179, 214, 0.55);
  color: #2b3a55;
  box-shadow: 0 2px 6px rgba(52, 86, 130, 0.08);
  letter-spacing: 0.3px;
}
#sora-hero .badge.ribbon-badge {
  background: #232836;
  border-color: #232836;
  color: #f4f7fb;
}
.dark #sora-hero .badge {
  background: rgba(22, 31, 52, 0.78);
  border-color: rgba(148, 179, 214, 0.32);
  color: #d6e4f5;
}
.dark #sora-hero .badge.ribbon-badge {
  background: rgba(138, 184, 232, 0.22);
  border-color: rgba(138, 184, 232, 0.45);
  color: #e8f2fc;
}

/* ───────────── Galgame 瓷白磨砂卡片与工作室面板 ───────────── */
.sora-card,
.sora-studio-panel,
.sora-dock {
  position: relative;
  border-radius: 20px !important;
  padding: 16px 18px 14px !important;
  border: 1px solid var(--sora-border) !important;
  background: transparent !important;
  isolation: isolate;
  box-shadow: var(--sora-card-shadow) !important;
  transition: border-color .22s ease, box-shadow .22s ease;
}
/* Keep fixed-position Gradio dropdowns out of the blur layer's containing block. */
.sora-card::before,
.sora-studio-panel::before,
.sora-dock::before {
  content: "";
  position: absolute;
  inset: 0;
  z-index: -1;
  border-radius: inherit;
  background: var(--sora-card-bg);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  pointer-events: none;
}

.sora-card:hover,
.sora-studio-panel:hover {
  border-color: rgba(91, 142, 201, 0.62) !important;
  box-shadow: var(--sora-card-hover-shadow) !important;
}
.sora-studio-panel > .styler,
.sora-dock > .styler,
.sora-card > .styler {
  background: transparent !important;
}
.sora-studio-panel {
  padding: 12px 14px 14px !important;
}
.sora-dock {
  margin-top: 16px !important;
  padding: 12px 16px 16px !important;
}

/* 蕾丝虚线子卡片 */
.sora-subcard {
  border-radius: 15px !important;
  padding: 12px 13px !important;
  margin-top: 8px !important;
  border: 1px dashed rgba(125, 160, 200, 0.52) !important;
  background: var(--sora-subcard-bg) !important;
}
.sora-subcard > .styler {
  background: transparent !important;
}

/* Galgame 章节铭牌标题 */
.sora-section {
  display: flex;
  align-items: center;
  gap: 8px;
  font-family: var(--sora-serif);
  font-weight: 700;
  font-size: 15.5px;
  padding: 0 2px 10px;
  color: var(--sora-ink);
  letter-spacing: 1px;
}
.sora-section .sec-sub {
  font-family: "Microsoft YaHei UI", sans-serif;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 1.4px;
  text-transform: uppercase;
  color: var(--sora-sky);
  opacity: 0.85;
  margin-left: 2px;
}
.sora-section::after {
  content: "";
  flex: 1;
  order: 1;
  height: 1px;
  margin-left: 6px;
  background: linear-gradient(90deg, rgba(91, 142, 201, 0.42) 0%, rgba(148, 179, 214, 0.10) 75%, transparent 100%);
}
.sora-hint {
  font-size: 12.5px;
  color: var(--sora-ink-soft);
  padding: 2px 4px 8px;
  line-height: 1.65;
}

/* ───────────── 全站控件铭牌标签（排除复选框标签与音频 SVG 图标 span） ───────────── */
.gradio-container span[data-testid="block-info"],
.gradio-container .block > label:not(:has(input[type="checkbox"])) > span:not(.icon):not(:empty):not(:has(svg)),
.gradio-container .block > .label-wrap > span:not(.icon):not(:empty):not(:has(svg)) {
  font-family: var(--sora-serif) !important;
  font-size: 12.5px !important;
  font-weight: 700 !important;
  letter-spacing: 0.6px !important;
  color: var(--sora-ink) !important;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.96) 0%, rgba(238, 245, 253, 0.92) 100%) !important;
  border: 1px solid rgba(148, 179, 214, 0.52) !important;
  border-radius: 8px !important;
  padding: 2.5px 9px !important;
  box-shadow: 0 1px 4px rgba(52, 86, 130, 0.06), inset 0 -1.5px 0 rgba(75, 124, 181, 0.35) !important;
}
.dark .gradio-container span[data-testid="block-info"],
.dark .gradio-container .block > label:not(:has(input[type="checkbox"])) > span:not(.icon):not(:empty):not(:has(svg)),
.dark .gradio-container .block > .label-wrap > span:not(.icon):not(:empty):not(:has(svg)) {
  color: #dce8f8 !important;
  background: linear-gradient(180deg, rgba(28, 38, 64, 0.94) 0%, rgba(20, 28, 48, 0.92) 100%) !important;
  border-color: rgba(138, 184, 232, 0.32) !important;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.24), inset 0 -1.5px 0 rgba(138, 184, 232, 0.42) !important;
}
/* 迷你分区插画横幅（用于声音效果 / 高级调音 / 自优化升级 / API 设置） */
.sora-tab-banner {
  position: relative;
  overflow: hidden;
  border-radius: 15px;
  padding: 13px 16px;
  margin-bottom: 10px;
  min-height: 74px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  border: 1px solid rgba(148, 179, 214, 0.48);
  box-shadow: 0 4px 14px rgba(52, 86, 130, 0.09), inset 0 1px 0 rgba(255, 255, 255, 0.85);
  background-size: 100% 100%, auto 116%;
  background-position: center, right -4px top 12%;
  background-repeat: no-repeat, no-repeat;
}
.dark .sora-tab-banner {
  border-color: rgba(138, 184, 232, 0.28);
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.32);
}
.sora-tab-banner .tb-title {
  font-family: var(--sora-serif);
  font-size: 15.5px;
  font-weight: 700;
  letter-spacing: 1px;
  color: var(--sora-ink);
}
.sora-tab-banner .tb-sub {
  font-size: 12px;
  color: var(--sora-ink-soft);
  margin-top: 2px;
  max-width: 68%;
  line-height: 1.45;
}

/* ───────────── 穹式按钮体系（白裙蕾丝双框 · 黑缎带蝴蝶结 · 苍穹丝绒） ───────────── */
.gradio-container button.primary,
.gradio-container button.secondary,
.gradio-container button.stop {
  position: relative;
  border-radius: 14px !important;
  font-weight: 700 !important;
  letter-spacing: 0.6px !important;
  transition: transform .16s ease, box-shadow .2s ease, border-color .18s ease, background .2s ease, filter .16s ease !important;
}

/* 常规按钮：对称白裙瓷感双框（纯净无杂点） */
.gradio-container button.secondary {
  color: #232b3e !important;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(242, 248, 254, 0.94) 100%) !important;
  border: 1px solid rgba(148, 179, 214, 0.60) !important;
  box-shadow:
    0 3px 10px rgba(52, 86, 130, 0.08),
    inset 0 0 0 2px rgba(255, 255, 255, 0.94) !important;
}
.gradio-container button.secondary:hover {
  transform: translateY(-1.5px) !important;
  color: #1c4474 !important;
  border-color: #4f85c4 !important;
  background: linear-gradient(180deg, #ffffff 0%, #e7f2fc 100%) !important;
  box-shadow:
    0 6px 16px rgba(74, 134, 200, 0.22),
    inset 0 0 0 2px rgba(255, 255, 255, 0.98) !important;
}
.gradio-container button.secondary:active {
  transform: translateY(0) scale(0.985) !important;
}
.dark .gradio-container button.secondary {
  color: #e2ecf9 !important;
  background: linear-gradient(180deg, rgba(30, 40, 66, 0.94) 0%, rgba(21, 29, 50, 0.92) 100%) !important;
  border: 1px solid rgba(138, 184, 232, 0.34) !important;
  box-shadow:
    0 4px 12px rgba(0, 0, 0, 0.30),
    inset 0 0 0 1.5px rgba(186, 210, 240, 0.09) !important;
}
.dark .gradio-container button.secondary:hover {
  color: #ffffff !important;
  border-color: rgba(168, 206, 246, 0.76) !important;
  background: linear-gradient(180deg, rgba(42, 56, 90, 0.96) 0%, rgba(30, 42, 72, 0.94) 100%) !important;
  box-shadow: 0 6px 16px rgba(91, 142, 201, 0.32) !important;
}

/* 次级主按钮（如按讨论生成/保存配置）：黑缎带晴空丝绒 + 微蕾丝点边 + 蝴蝶结暗纹 */
.gradio-container button.primary {
  color: #ffffff !important;
  border: 1px solid rgba(255, 255, 255, 0.56) !important;
  background:
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='22' height='16' viewBox='0 0 36 26' fill='none'%3E%3Cpath d='M18 11C13 5 4 4.5 3.5 10.5C3 15.5 10.5 16 16.5 13M18 11C23 5 32 4.5 32.5 10.5C33 15.5 25.5 16 19.5 13' fill='%23ffffff' fill-opacity='0.24'/%3E%3Ccircle cx='18' cy='12' r='2.5' fill='%23ffffff' fill-opacity='0.34'/%3E%3C/svg%3E") right 10px center / 20px 15px no-repeat,
    radial-gradient(circle, rgba(255, 255, 255, 0.32) 0.9px, transparent 1px) left bottom 3px / 10px 4px repeat-x,
    linear-gradient(110deg, #242c40 0%, #4271a8 56%, #6da3db 100%) !important;
  box-shadow:
    0 5px 16px rgba(45, 76, 120, 0.28),
    inset 0 0 0 2px rgba(255, 255, 255, 0.16) !important;
}
.gradio-container button.primary:hover {
  transform: translateY(-1.5px) !important;
  filter: brightness(1.07) !important;
  box-shadow:
    0 9px 22px rgba(53, 94, 145, 0.38),
    inset 0 0 0 2px rgba(255, 255, 255, 0.28) !important;
}
.gradio-container button.primary:active {
  transform: translateY(0) scale(0.985) !important;
}

/* 警示/清空按钮：对称蔷薇蕾丝卡扣 */
.gradio-container button.stop {
  color: #9e3250 !important;
  background:
    radial-gradient(circle, rgba(201, 85, 117, 0.26) 0.9px, transparent 1px) left bottom 3px / 10px 4px repeat-x,
    linear-gradient(180deg, rgba(255, 247, 249, 0.98) 0%, rgba(252, 235, 240, 0.94) 100%) !important;
  border: 1px solid rgba(214, 126, 152, 0.56) !important;
  box-shadow: 0 3px 10px rgba(190, 75, 108, 0.10), inset 0 0 0 2px rgba(255, 255, 255, 0.92) !important;
}
.gradio-container button.stop:hover {
  transform: translateY(-1.5px) !important;
  color: #ffffff !important;
  border-color: transparent !important;
  background: linear-gradient(110deg, #c95575 0%, #a83757 100%) !important;
  box-shadow: 0 7px 18px rgba(182, 62, 96, 0.28) !important;
}
.dark .gradio-container button.stop {
  color: #f3b0c2 !important;
  background:
    radial-gradient(circle, rgba(226, 157, 178, 0.28) 0.9px, transparent 1px) left bottom 3px / 10px 4px repeat-x,
    linear-gradient(180deg, rgba(66, 26, 40, 0.90) 0%, rgba(50, 20, 32, 0.88) 100%) !important;
  border: 1px solid rgba(214, 126, 152, 0.44) !important;
}
.dark .gradio-container button.stop:hover {
  color: #ffffff !important;
  background: linear-gradient(110deg, #c95575 0%, #9e3150 100%) !important;
}

/* ───────────── 核心主生成按钮（穹 · 白裙蕾丝 × 黑缎带蝴蝶结 × 黑兔主控） ───────────── */
#sora-gen-btn {
  position: relative;
  overflow: hidden;
  min-height: 62px !important;
  font-family: var(--sora-serif) !important;
  font-size: 18.5px !important;
  font-weight: 700 !important;
  letter-spacing: 2.4px !important;
  border-radius: 18px !important;
  border: 1.5px solid rgba(255, 255, 255, 0.78) !important;
  color: #ffffff !important;
  background:
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='28' height='28' viewBox='0 0 24 24' fill='none'%3E%3Cpath d='M8.5 10.5C7 5.5 7.5 2.5 9.5 2.8C11.2 3.1 11.4 6.8 11 10' fill='%23ffffff' fill-opacity='0.24'/%3E%3Cpath d='M15.5 10.5C17 5.5 16.5 2.5 14.5 2.8C12.8 3.1 12.6 6.8 13 10' fill='%23ffffff' fill-opacity='0.24'/%3E%3Ccircle cx='12' cy='14.8' r='6.2' fill='%23ffffff' fill-opacity='0.24'/%3E%3C/svg%3E") left 22px center / 26px 26px no-repeat,
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='36' height='26' viewBox='0 0 36 26' fill='none'%3E%3Cpath d='M18 11C13 5 4 4.5 3.5 10.5C3 15.5 10.5 16 16.5 13' fill='%23ffffff' fill-opacity='0.26'/%3E%3Cpath d='M18 11C23 5 32 4.5 32.5 10.5C33 15.5 25.5 16 19.5 13' fill='%23ffffff' fill-opacity='0.26'/%3E%3Cpath d='M16 13L11.5 23.5L15.5 21.5L17.5 14M20 13L24.5 23.5L20.5 21.5L18.5 14' fill='%23ffffff' fill-opacity='0.20'/%3E%3Ccircle cx='18' cy='12' r='2.6' fill='%23ffffff' fill-opacity='0.36'/%3E%3C/svg%3E") right 22px center / 34px 24px no-repeat,
    radial-gradient(circle, rgba(255, 255, 255, 0.36) 1px, transparent 1.2px) left top 4px / 12px 6px repeat-x,
    radial-gradient(circle, rgba(255, 255, 255, 0.30) 1px, transparent 1.2px) left bottom 4px / 12px 6px repeat-x,
    linear-gradient(108deg, #1f2536 0%, #345a8f 48%, #6096cf 100%) !important;
  box-shadow:
    0 10px 26px rgba(35, 62, 104, 0.34),
    0 0 0 1px rgba(31, 37, 54, 0.22),
    inset 0 0 0 3px rgba(255, 255, 255, 0.14) !important;
  transition: transform .18s ease, box-shadow .22s ease, filter .18s ease !important;
  margin: 10px 0 !important;
}
.dark #sora-gen-btn {
  border-color: rgba(186, 214, 245, 0.48) !important;
  box-shadow:
    0 12px 30px rgba(8, 16, 36, 0.56),
    inset 0 0 0 3px rgba(186, 214, 245, 0.12) !important;
}
#sora-gen-btn::after {
  content: "";
  position: absolute;
  top: -25%;
  left: -80%;
  width: 44%;
  height: 150%;
  background: linear-gradient(105deg, transparent 0%, rgba(255, 255, 255, 0.42) 50%, transparent 100%);
  transform: skewX(-22deg);
  transition: left .62s ease;
  pointer-events: none;
}
#sora-gen-btn:hover::after { left: 138%; }
#sora-gen-btn:hover {
  transform: translateY(-2px) !important;
  filter: brightness(1.08) !important;
  box-shadow:
    0 14px 34px rgba(55, 93, 145, 0.44),
    inset 0 0 0 3px rgba(255, 255, 255, 0.26) !important;
}
#sora-gen-btn:active { transform: translateY(0) scale(0.99) !important; }

/* 4 个穹式情境预设卡片按钮（ComfyUI 定制立绘卡面） */
.sora-preset-row {
  gap: 8px !important;
  margin-bottom: 8px !important;
}
.sora-preset-row button {
  min-height: 74px !important;
  font-family: var(--sora-serif) !important;
  font-size: 14px !important;
  font-weight: 700 !important;
  letter-spacing: 0.8px !important;
  border-radius: 15px !important;
  justify-content: flex-start !important;
  text-align: left !important;
  padding: 10px 14px !important;
  overflow: hidden !important;
}
.sora-preset-row button:hover {
  transform: translateY(-2px) !important;
  filter: brightness(1.03) !important;
}

/* 完整展示 Q 版穹看板娘贴纸（行内右对齐，防父容器裁切头部） */
.sora-section-chibi {
  min-height: 52px;
  padding-top: 2px !important;
  padding-bottom: 6px !important;
}
.sora-chibi-float {
  order: 2;
  width: 54px;
  height: 54px;
  margin-left: 8px;
  flex-shrink: 0;
  object-fit: contain;
  pointer-events: none;
  filter: drop-shadow(0 3px 6px rgba(35, 52, 86, 0.20));
}
.gradio-container button.sora-preset-daily {
  border: 1px solid rgba(85, 137, 196, 0.58) !important;
  box-shadow: 0 3px 10px rgba(85, 137, 196, 0.10), inset 0 -2.5px 0 #5589c4 !important;
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(232, 244, 254, 0.94) 100%) !important;
}
.dark .gradio-container button.sora-preset-daily {
  border-color: rgba(125, 180, 235, 0.44) !important;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.28), inset 0 -2.5px 0 #7db4eb !important;
  background: linear-gradient(135deg, rgba(28, 42, 68, 0.92) 0%, rgba(22, 34, 56, 0.90) 100%) !important;
}
.gradio-container button.sora-preset-clingy {
  border: 1px solid rgba(212, 132, 157, 0.58) !important;
  box-shadow: 0 3px 10px rgba(212, 132, 157, 0.12), inset 0 -2.5px 0 #d4849d !important;
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(253, 241, 246, 0.94) 100%) !important;
}
.dark .gradio-container button.sora-preset-clingy {
  border-color: rgba(229, 164, 184, 0.44) !important;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.28), inset 0 -2.5px 0 #e5a4b8 !important;
  background: linear-gradient(135deg, rgba(54, 30, 48, 0.90) 0%, rgba(36, 24, 42, 0.90) 100%) !important;
}
.gradio-container button.sora-preset-tsun {
  border: 1px solid rgba(55, 65, 88, 0.45) !important;
  box-shadow: 0 3px 10px rgba(35, 40, 54, 0.10), inset 0 -2.5px 0 #262c3d !important;
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(238, 241, 247, 0.94) 100%) !important;
}
.dark .gradio-container button.sora-preset-tsun {
  border-color: rgba(203, 213, 225, 0.36) !important;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.28), inset 0 -2.5px 0 #cbd5e1 !important;
  background: linear-gradient(135deg, rgba(32, 37, 52, 0.94) 0%, rgba(22, 26, 38, 0.92) 100%) !important;
}
.gradio-container button.sora-preset-asmr {
  border: 1px solid rgba(168, 210, 255, 0.55) !important;
  box-shadow: 0 5px 16px rgba(30, 52, 92, 0.30), inset 0 -2.5px 0 #a8d2ff !important;
  background: linear-gradient(125deg, #1b243d 0%, #2d4b7c 60%, #4b78b3 100%) !important;
  color: #ffffff !important;
}

/* ───────────── 分段黑缎带书签导航 Tab（含蕾丝底边与蝴蝶结选中饰） ───────────── */
.sora-tabs > .tab-nav {
  border-bottom: none !important;
  gap: 4px !important;
  padding: 5px 6px 8px !important;
  border-radius: 15px !important;
  background:
    radial-gradient(circle, rgba(91, 142, 201, 0.30) 1px, transparent 1.1px) left bottom 3px / 11px 4px repeat-x,
    rgba(238, 245, 253, 0.84) !important;
  border: 1px solid rgba(148, 179, 214, 0.45) !important;
  margin-bottom: 10px !important;
}
.dark .sora-tabs > .tab-nav {
  background:
    radial-gradient(circle, rgba(138, 184, 232, 0.28) 1px, transparent 1.1px) left bottom 3px / 11px 4px repeat-x,
    rgba(12, 17, 31, 0.84) !important;
  border-color: rgba(148, 179, 214, 0.24) !important;
}
.sora-tabs > .tab-nav button {
  font-family: var(--sora-serif) !important;
  font-weight: 700 !important;
  font-size: 12.5px !important;
  letter-spacing: 0.2px !important;
  padding: 6px 8px !important;
  border-radius: 10px !important;
  border: 1px solid transparent !important;
  color: var(--sora-ink-soft) !important;
  transition: all .18s ease !important;
}
.sora-tabs > .tab-nav button:hover {
  color: var(--sora-ink) !important;
  border-color: rgba(148, 179, 214, 0.45) !important;
  background: rgba(255, 255, 255, 0.78) !important;
}
.dark .sora-tabs > .tab-nav button:hover {
  border-color: rgba(138, 184, 232, 0.28) !important;
  background: rgba(138, 184, 232, 0.14) !important;
}
.sora-tabs > .tab-nav button.selected {
  color: #ffffff !important;
  border-color: rgba(255, 255, 255, 0.40) !important;
  background: linear-gradient(108deg, #212738 0%, #3d6ba3 68%, #6396ce 100%) !important;
  box-shadow:
    0 4px 12px rgba(35, 62, 104, 0.30),
    inset 0 -2px 0 rgba(186, 220, 255, 0.82) !important;
}
.sora-tabs > .tabitem { padding: 4px 2px 2px !important; border: none !important; }

/* ───────────── 全站单选、滑动条（珍珠黑缎带滑块）、复选框与折叠面板 ───────────── */
.sora-inline-bar {
  align-items: center !important;
  gap: 10px !important;
  margin-top: 4px !important;
}
.sora-align-end {
  align-items: flex-end !important;
}
.sora-radio-stack .wrap { flex-direction: column !important; align-items: stretch !important; gap: 6px !important; }

/* 全局统一所有单选按钮为穹式瓷白 / 黑缎带晴空选中态 */
.gradio-container .wrap label:has(input[type="radio"]):not(.selected):not(:has(input:checked)) {
  border-radius: 11px !important;
  border: 1px solid rgba(148, 179, 214, 0.42) !important;
  background: var(--sora-card-bg) !important;
  color: var(--sora-ink) !important;
  transition: border-color .16s ease, background .16s ease, color .16s ease;
}
.gradio-container .wrap label:has(input[type="radio"]):not(.selected):not(:has(input:checked)):hover {
  border-color: rgba(85, 137, 196, 0.78) !important;
}
.gradio-container .wrap label.selected:has(input[type="radio"]),
.gradio-container .wrap label:has(input[type="radio"]:checked) {
  border-radius: 11px !important;
  background: linear-gradient(108deg, #232a3d 0%, #4472a8 70%, #6396ce 100%) !important;
  border: 1px solid rgba(255, 255, 255, 0.36) !important;
  color: #ffffff !important;
  box-shadow: 0 4px 12px rgba(42, 72, 114, 0.24) !important;
}
.gradio-container .wrap label.selected:has(input[type="radio"]) span,
.gradio-container .wrap label:has(input[type="radio"]:checked) span {
  color: #ffffff !important;
}

/* 全站滑动条：白裙珍珠圆钮 + 黑缎带细环 */
.gradio-container input[type="range"] {
  accent-color: #4b7cb5 !important;
}
.gradio-container input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none !important;
  width: 17px !important;
  height: 17px !important;
  border-radius: 50% !important;
  background: radial-gradient(circle at 35% 35%, #ffffff 0%, #eef5fd 75%, #d6e6f7 100%) !important;
  border: 2px solid #262d3f !important;
  box-shadow: 0 2px 6px rgba(35, 48, 74, 0.28), 0 0 0 2px rgba(255, 255, 255, 0.75) !important;
  cursor: pointer;
  transition: transform .15s ease, box-shadow .18s ease, border-color .18s ease;
}
.gradio-container input[type="range"]::-webkit-slider-thumb:hover {
  transform: scale(1.14);
  border-color: #4b7cb5 !important;
  box-shadow: 0 3px 10px rgba(75, 124, 181, 0.42), 0 0 0 3px rgba(138, 184, 232, 0.32) !important;
}
.dark .gradio-container input[type="range"]::-webkit-slider-thumb {
  background: radial-gradient(circle at 35% 35%, #ffffff 0%, #dbeafe 80%) !important;
  border-color: #6ea4dc !important;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.48), 0 0 0 2px rgba(138, 184, 232, 0.22) !important;
}

textarea:focus, input:focus {
  box-shadow: 0 0 0 3px rgba(91, 142, 201, 0.26) !important;
}

/* 折叠面板：丝带信笺风格 */
.sora-acc {
  border-radius: 14px !important;
  border: 1px solid rgba(148, 179, 214, 0.48) !important;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.88) 0%, rgba(242, 248, 254, 0.82) 100%) !important;
  box-shadow: 0 2px 8px rgba(52, 86, 130, 0.06), inset 0 1px 0 rgba(255, 255, 255, 0.95) !important;
  margin-top: 8px !important;
}
.dark .sora-acc {
  background: linear-gradient(180deg, rgba(22, 30, 50, 0.82) 0%, rgba(16, 22, 38, 0.80) 100%) !important;
  border-color: rgba(148, 179, 214, 0.24) !important;
  box-shadow: 0 3px 10px rgba(0, 0, 0, 0.28) !important;
}
.sora-acc > button {
  font-family: var(--sora-serif) !important;
  font-weight: 700 !important;
  letter-spacing: 0.6px !important;
  color: var(--sora-ink) !important;
}

/* 导演聊天区：Galgame 对话气泡质感 */
.sora-chat {
  border-radius: 16px !important;
  border: 1px solid rgba(148, 179, 214, 0.48) !important;
  background: rgba(249, 252, 255, 0.84) !important;
}
.dark .sora-chat {
  background: rgba(13, 18, 32, 0.84) !important;
  border-color: rgba(148, 179, 214, 0.24) !important;
}

/* 音频播放器：苍穹八音盒 + 淡雅黑兔水印 */
.sora-audio-box {
  border-radius: 15px !important;
  border: 1px solid rgba(148, 179, 214, 0.52) !important;
  background:
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='56' height='56' viewBox='0 0 24 24' fill='none'%3E%3Cpath d='M8.5 10.5C7 5.5 7.5 2.5 9.5 2.8C11.2 3.1 11.4 6.8 11 10M15.5 10.5C17 5.5 16.5 2.5 14.5 2.8C12.8 3.1 12.6 6.8 13 10' fill='%234b7cb5' fill-opacity='0.08'/%3E%3Ccircle cx='12' cy='14.8' r='6.2' fill='%234b7cb5' fill-opacity='0.08'/%3E%3C/svg%3E") right 14px bottom 6px / 52px 52px no-repeat,
    linear-gradient(180deg, rgba(255, 255, 255, 0.94) 0%, rgba(238, 246, 254, 0.90) 100%) !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.95) !important;
}
.dark .sora-audio-box {
  background:
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='56' height='56' viewBox='0 0 24 24' fill='none'%3E%3Cpath d='M8.5 10.5C7 5.5 7.5 2.5 9.5 2.8C11.2 3.1 11.4 6.8 11 10M15.5 10.5C17 5.5 16.5 2.5 14.5 2.8C12.8 3.1 12.6 6.8 13 10' fill='%238ab8e8' fill-opacity='0.10'/%3E%3Ccircle cx='12' cy='14.8' r='6.2' fill='%238ab8e8' fill-opacity='0.10'/%3E%3C/svg%3E") right 14px bottom 6px / 52px 52px no-repeat,
    linear-gradient(180deg, rgba(22, 30, 50, 0.90) 0%, rgba(16, 22, 38, 0.90) 100%) !important;
  border-color: rgba(148, 179, 214, 0.28) !important;
}

/* 浅色/深色未选中单选圆圈与复选框边界清晰化 + 单行输入框去拖拽角标 */
.gradio-container input[type="radio"]:not(:checked),
.gradio-container input[type="checkbox"]:not(:checked) {
  border: 1.5px solid rgba(112, 140, 178, 0.62) !important;
  background-color: rgba(255, 255, 255, 0.96) !important;
}
.dark .gradio-container input[type="radio"]:not(:checked),
.dark .gradio-container input[type="checkbox"]:not(:checked) {
  border-color: rgba(148, 179, 214, 0.45) !important;
  background-color: rgba(18, 25, 44, 0.92) !important;
}
.gradio-container textarea[rows="1"] {
  resize: none !important;
}

/* 音频库表格：苍穹晴空表头、禁短列折行与柔光悬停 */
.sora-table thead th {
  background: linear-gradient(180deg, #eef6fd 0%, #e1eefb 100%) !important;
  color: #24314a !important;
  font-family: var(--sora-serif) !important;
  font-weight: 700 !important;
  white-space: nowrap !important;
  border-bottom: 1.5px solid rgba(148, 179, 214, 0.55) !important;
}
.sora-table tbody td:nth-child(1),
.sora-table tbody td:nth-child(2),
.sora-table tbody td:nth-child(3) {
  white-space: nowrap !important;
}
.dark .sora-table thead th {
  background: linear-gradient(180deg, #1b2540 0%, #141c32 100%) !important;
  color: #dce8f8 !important;
  border-bottom-color: rgba(138, 184, 232, 0.30) !important;
}
.sora-table tbody tr:hover td {
  background: rgba(91, 142, 201, 0.10) !important;
}
.dark .sora-table tbody tr:hover td {
  background: rgba(138, 184, 232, 0.12) !important;
}
.dark .sora-audio-box ::-webkit-scrollbar-track {
  background: rgba(16, 22, 38, 0.85) !important;
}
.dark .sora-audio-box ::-webkit-scrollbar-thumb {
  background: rgba(138, 184, 232, 0.45) !important;
}

/* 锁定标签片段卡片（适配浅色/深色） */
.sora-pill-empty {
  padding: 14px;
  border: 1.5px dashed rgba(148, 179, 214, 0.55);
  border-radius: 13px;
  color: var(--sora-ink-soft);
  font-size: 13px;
  text-align: center;
  background: rgba(255, 255, 255, 0.55);
}
.dark .sora-pill-empty {
  background: rgba(16, 22, 38, 0.55);
  border-color: rgba(148, 179, 214, 0.28);
}
.sora-pill-row {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
  padding: 9px 12px;
  margin-bottom: 6px;
  background: rgba(255, 255, 255, 0.90);
  border: 1px solid rgba(148, 179, 214, 0.45);
  border-left: 3px solid #232836;
  border-radius: 12px;
  box-shadow: 0 2px 6px rgba(52, 86, 130, 0.06);
}
.dark .sora-pill-row {
  background: rgba(22, 30, 50, 0.86);
  border-color: rgba(148, 179, 214, 0.24);
  border-left-color: #8ab8e8;
}
.sora-pill-idx {
  color: var(--sora-sky);
  font-weight: 800;
  font-size: 12px;
  min-width: 18px;
}
.sora-pill-meta {
  background: var(--sora-sky-soft);
  color: var(--sora-ink-soft);
  padding: 2.5px 8px;
  border-radius: 999px;
  font-size: 11.5px;
  font-weight: 600;
}
.sora-pill-text {
  color: var(--sora-ink);
  font-size: 13.5px;
  margin-left: 4px;
  flex: 1;
  min-width: 120px;
}

/* ───────────── 苍穹晴空进度条（附黑兔剪影） ───────────── */
.sora-prog-card {
  border-radius: 15px;
  padding: 10px 14px;
  margin: 4px 0 10px;
  background: rgba(255, 255, 255, 0.90);
  border: 1px solid rgba(148, 179, 214, 0.52);
  box-shadow: 0 4px 14px rgba(52, 86, 130, 0.10);
}
.dark .sora-prog-card {
  background: rgba(18, 25, 44, 0.90);
  border-color: rgba(138, 184, 232, 0.30);
}
.sora-prog-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 13px;
  font-weight: 700;
  color: var(--sora-ink);
  margin-bottom: 7px;
}
.sora-prog-title {
  display: inline-flex;
  align-items: center;
  gap: 7px;
}
.sora-prog-track {
  width: 100%;
  height: 8px;
  border-radius: 999px;
  background: rgba(214, 228, 245, 0.85);
  overflow: hidden;
}
.dark .sora-prog-track { background: rgba(30, 42, 68, 0.92); }
.sora-prog-fill {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, #232a3d 0%, #4b7cb5 55%, #84b6e8 100%);
  transition: width .22s ease;
}
.gradio-container .progress-bar > .progress,
.gradio-container .progress-level .progress-bar > div {
  background: linear-gradient(90deg, #232a3d 0%, #4b7cb5 55%, #84b6e8 100%) !important;
  border-radius: 999px !important;
}
.gradio-container .wrap .eta-bar,
.gradio-container .progress-text {
  font-weight: 700 !important;
  color: var(--sora-sky-deep) !important;
}
.dark .gradio-container .wrap .eta-bar,
.dark .gradio-container .progress-text {
  color: #9bc5ee !important;
}
""" + _BG_WALLPAPER_CSS

_BLACK_RIBBON_SVG = """<svg class="crest-bow" viewBox="0 0 36 26" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M18 11C13 5 4 4.5 3.5 10.5C3 15.5 10.5 16 16.5 13" fill="#232836" stroke="#ffffff" stroke-width="1.3"/>
  <path d="M18 11C23 5 32 4.5 32.5 10.5C33 15.5 25.5 16 19.5 13" fill="#232836" stroke="#ffffff" stroke-width="1.3"/>
  <path d="M16 13L11.5 23.5L15.5 21.5L17.5 14" fill="#2b3142"/>
  <path d="M20 13L24.5 23.5L20.5 21.5L18.5 14" fill="#2b3142"/>
  <circle cx="18" cy="12" r="2.8" fill="#1b202c" stroke="#e2ecf8" stroke-width="1.1"/>
</svg>"""

_KURO_USAGI_ICON_SVG = """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;">
  <path d="M8.5 10.5C7 5.5 7.5 2.5 9.5 2.8C11.2 3.1 11.4 6.8 11 10" fill="currentColor" opacity="0.88"/>
  <path d="M15.5 10.5C17 5.5 16.5 2.5 14.5 2.8C12.8 3.1 12.6 6.8 13 10" fill="currentColor" opacity="0.88"/>
  <circle cx="12" cy="14.8" r="6.2" fill="currentColor"/>
  <path d="M9.6 14.2L10.8 15.4M10.8 14.2L9.6 15.4M13.2 14.2L14.4 15.4M14.4 14.2L13.2 15.4" stroke="#ffffff" stroke-width="1.1" stroke-linecap="round"/>
</svg>"""

_HERO_CREST_HTML = (
    f'<img src="{_AVATAR_URI}" alt="春日野穹" />'
    if _AVATAR_URI
    else _KURO_USAGI_ICON_SVG
)

UI_HERO = f"""
<div id="sora-hero">
  <div class="hero-main">
    <div class="sora-crest-wrap" aria-hidden="true">
      <div class="sora-crest">
        {_HERO_CREST_HTML}
      </div>
      {_BLACK_RIBBON_SVG}
    </div>
    <div class="hero-copy">
      <div class="jp-kicker">✦ YOSUGA NO SORA · KASUGANO SORA VOICE STUDIO</div>
      <div class="title">春日野 穹<span class="title-sep">·</span>表演向语音合成</div>
      <div class="subtitle">AI 导演潜台词编排 · 63 档声学参考匹配 · 贴耳轻语（ASMR）声场</div>
      <div class="badges">
        <span class="badge ribbon-badge">🎀 苍穹声场</span>
        <span class="badge">GPT · ep15-ft1</span>
        <span class="badge">SoVITS · ep25</span>
        <span class="badge">🌙 ASMR · ep5</span>
      </div>
    </div>
  </div>
</div>
"""
