import json
from synthesize import engine,render
from performance import ROOT,plan_with_ollama,torch

def main():
    text="約束したのに……でも、帰ってきてくれてよかった。"
    context="前半の『約束したのに』は、約束を破った相手への怒りと悲しみ。後半は無事に帰った相手への優しさ。『でも』の前で短く間を置く。"
    auto=plan_with_ollama(text,context)
    cut=text.index("でも")
    a=[dict(start=0,end=cut,primary="anger",secondary="hurt",intensity=.85,pause_s=.4),
       dict(start=cut,end=len(text),primary="tender",secondary=None,intensity=.35,pause_s=0.)]
    b=[dict(start=0,end=cut,primary="tender",secondary=None,intensity=.35,pause_s=.4),
       dict(start=cut,end=len(text),primary="anger",secondary="hurt",intensity=.85,pause_s=0.)]
    models=ROOT/"results"
    tts=engine(models/"phrase-adapter-epoch2.pt")
    cases={}
    for name,plan,enabled in (
        ("自动规划",auto,True),("前责怪后放轻",a,True),("前放轻后责怪",b,True),
        ("无条件基线",a,False),
    ):
        cases[name]={"plan":plan,**render(tts,text,plan,models/f"{name}.wav",enabled)}
    (models/"同句对照.json").write_text(json.dumps({"text":text,"cases":cases},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(cases,ensure_ascii=False))
    assert cases["前责怪后放轻"]["semantic_sha256"]!=cases["无条件基线"]["semantic_sha256"]
    assert cases["前责怪后放轻"]["semantic_sha256"]!=cases["前放轻后责怪"]["semantic_sha256"]

if __name__=="__main__":
    main()
