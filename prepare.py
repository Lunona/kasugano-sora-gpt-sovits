import os,sys,runpy
from pathlib import Path
root=Path(__file__).resolve().parent
app=Path('F:/GPT-SoVITS/app')
pre=app/'GPT_SoVITS/pretrained_models'
data=root.parent/'春日野穹_语音数据'
os.environ.update({'HF_HOME':'F:/GPT-SoVITS/cache/huggingface','TEMP':str(root/'tmp'),'TMP':str(root/'tmp'),'PYTHONUTF8':'1','version':'v2','USE_LIBUV':'0','hz':'25hz','inp_text':str(data/'GPT-SoVITS_候选相对路径.list'),'inp_wav_dir':str(data/'音频_WAV'),'exp_name':'sora-performance','opt_dir':str(root/'prep'),'i_part':'0','all_parts':'1','is_half':'True','bert_pretrained_dir':str(pre/'chinese-roberta-wwm-ext-large'),'cnhubert_base_dir':str(pre/'chinese-hubert-base'),'pretrained_s2G':str(pre/'gsv-v2final-pretrained/s2G2333k.pth'),'s2config_path':str(app/'GPT_SoVITS/configs/s2.json')})
os.environ['PATH']=str(app/'runtime')+os.pathsep+os.environ['PATH']
sys.path[:0]=[str(app/'GPT_SoVITS'),str(app)]
os.chdir(root)
for stage in sys.argv[1:]:
 runpy.run_path(str(app/'GPT_SoVITS/prepare_datasets'/stage),run_name='__main__')
