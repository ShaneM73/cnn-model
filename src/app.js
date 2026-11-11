(function(){
"use strict";
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const M=JSON.parse($("#model").textContent);
const CLS=["Plus","Cross","Square","Triangle","Circle"], GLY=["+","×","□","△","○"];
const reduced=matchMedia("(prefers-reduced-motion: reduce)").matches;
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const fmt=(v,d=2)=>(v<0?"−":"")+Math.abs(v).toFixed(d);
const pct=v=>(v*100>=99.95&&v<1?"99.9":(v*100).toFixed(v>=0.1||v===0?0:1))+"%";

/* ================= The trained network (weights from model.json) ================= */
const WT=M.weights;
const W1=WT.W1,B1=WT.b1,W2=WT.W2,B2=WT.b2,W3=WT.W3,B3=WT.b3,WD=WT.Wd,BD=WT.bd;
function conv(X,C,H,W,b,F){ // zero padding, 3x3, stride 1. W layout [F][C][3][3]
  const HH=H*H,Z=new Float32Array(F*HH);
  for(let f=0;f<F;f++){const zb=f*HH;
    for(let c=0;c<C;c++){const xb=c*HH,wb=(f*C+c)*9;
      for(let r=0;r<H;r++)for(let q=0;q<H;q++){let acc=0;
        for(let dr=-1;dr<=1;dr++){const rr=r+dr;if(rr<0||rr>=H)continue;
          for(let dc=-1;dc<=1;dc++){const cc=q+dc;if(cc<0||cc>=H)continue;acc+=W[wb+(dr+1)*3+dc+1]*X[xb+rr*H+cc];}}
        Z[zb+r*H+q]+=acc;}}
    for(let i=0;i<HH;i++)Z[zb+i]+=b[f];}
  return Z;}
const relu=Z=>Z.map(v=>v>0?v:0);
function pool(X,C,H){const h=H/2,Y=new Float32Array(C*h*h);
  for(let c=0;c<C;c++)for(let r=0;r<h;r++)for(let q=0;q<h;q++){const b=c*H*H;
    Y[c*h*h+r*h+q]=Math.max(X[b+2*r*H+2*q],X[b+2*r*H+2*q+1],X[b+(2*r+1)*H+2*q],X[b+(2*r+1)*H+2*q+1]);}
  return Y;}
function zeroCh(A,HH,set){if(set)set.forEach(k=>A.fill(0,k*HH,(k+1)*HH));return A;}
function forward(x,off){ // off: optional {1:Set,2:Set,3:Set} of channels to knock out
  const z1=conv(x,1,28,W1,B1,8),a1=zeroCh(relu(z1),784,off&&off[1]),p1=pool(a1,8,28);
  const z2=conv(p1,8,14,W2,B2,12),a2=zeroCh(relu(z2),196,off&&off[2]),p2=pool(a2,12,14);
  const z3=conv(p2,12,7,W3,B3,16),a3=zeroCh(relu(z3),49,off&&off[3]);
  const g=new Float32Array(16),gi=new Int32Array(16);for(let k=0;k<16;k++){let m=-1,mi=0;for(let i=0;i<49;i++)if(a3[k*49+i]>m){m=a3[k*49+i];mi=i;}g[k]=m;gi[k]=mi;}
  const lg=[0,1,2,3,4].map(c=>{let s=BD[c];for(let k=0;k<16;k++)s+=WD[c*16+k]*g[k];return s;});
  const mx=Math.max(...lg),ex=lg.map(v=>Math.exp(v-mx)),se=ex.reduce((a,b)=>a+b,0);
  return {x,z1,a1,p1,z2,a2,p2,z3,a3,g,gi,lg,prob:ex.map(v=>v/se),win:lg.indexOf(mx)};
}

/* ================= Groups (what is drawn in 3D) ================= */
const SZ=6;
const GR=[
  {key:"input",n:1,H:28,x0:0,gap:0,rf:1,label:"Input",th:.26},
  {key:"conv1",n:8,H:28,x0:3.6,gap:.55,rf:3,label:"Conv 1 · 8 maps",th:.16},
  {key:"pool1",n:8,H:14,x0:10.6,gap:.55,rf:4,label:"Pool · 14×14",th:.16},
  {key:"conv2",n:12,H:14,x0:17.6,gap:.45,rf:8,label:"Conv 2 · 12 maps",th:.16},
  {key:"pool2",n:12,H:7,x0:25.6,gap:.45,rf:10,label:"Pool · 7×7",th:.16},
  {key:"conv3",n:16,H:7,x0:33.4,gap:.38,rf:18,label:"Conv 3 · 16 maps",th:.16},
  {key:"avg",n:16,H:1,x0:42.6,rf:28,label:"Global max"},
  {key:"out",n:5,H:1,x0:45.6,rf:28,label:""}];
const NG=GR.length;
GR.forEach(g=>{if(g.H>1){g.x1=g.x0+(g.n-1)*g.gap;g.cs=SZ/g.H;}else g.x1=g.x0;g.xm=(g.x0+g.x1)/2;});
GR[7].xm=GR[7].x0+2.6;GR[6].lblY=4.5;GR[7].lblY=3.7;
const OPS=[null,"conv","pool","conv","pool","conv"];
function groupData(gi){if(!NET)return null;return [NET.x,NET.a1,NET.p1,NET.a2,NET.p2,NET.a3][gi];}
function rfRanges(gi,r,c){ // receptive field of unit (r,c) of group gi, in every earlier group's grid
  const R={};let a=r,b=r,e=c,f=c;R[gi]=[a,b,e,f];
  for(let k=gi;k>=1;k--){if(OPS[k]==="conv"){a--;b++;e--;f++;}else{a*=2;b=b*2+1;e*=2;f=f*2+1;}R[k-1]=[a,b,e,f];}
  return R;}
const cellPos=(gi,ch,r,c)=>{const g=GR[gi];return [g.x0+ch*g.gap,SZ/2-(r+.5)*g.cs,(c+.5)*g.cs-SZ/2];};

/* ================= Drawing pad ================= */
const pad=$("#pad"),pctx=pad.getContext("2d"),padOv=$("#padOv");
const inkC=document.createElement("canvas");inkC.width=inkC.height=280;const ictx=inkC.getContext("2d",{willReadFrequently:true});
let strokes=[],cur=null;
function strokePath(ctx,s){ctx.lineWidth=s.w;ctx.lineCap="round";ctx.lineJoin="round";ctx.beginPath();
  s.pts.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));if(s.pts.length===1)ctx.lineTo(s.pts[0][0]+.01,s.pts[0][1]);ctx.stroke();}
function redrawPad(){
  pctx.fillStyle=CS.bg;pctx.fillRect(0,0,280,280);pctx.strokeStyle=CS.fg;strokes.forEach(s=>strokePath(pctx,s));
  if(S.stim){for(let i=0;i<784;i++){const v=S.stim[i];if(v>.01){pctx.fillStyle=hexA(CS.fg,v);pctx.fillRect((i%28)*10,((i/28)|0)*10,10,10);}}}
  $("#stimNote").hidden=!S.stim;
  ictx.fillStyle="#000";ictx.fillRect(0,0,280,280);ictx.strokeStyle="#fff";strokes.forEach(s=>strokePath(ictx,s));
  $("#padHint").hidden=strokes.length>0||!!S.stim;}
function normalize(){ // same recipe as the training data: bbox -> fit 20px -> centre at 14 -> area average
  const d=ictx.getImageData(0,0,280,280).data,out=new Float32Array(784);
  let x0=1e9,x1=-1,y0=1e9,y1=-1;
  for(let y=0;y<280;y++)for(let x=0;x<280;x++)if(d[(y*280+x)*4]>25.5){if(x<x0)x0=x;if(x>x1)x1=x;if(y<y0)y0=y;if(y>y1)y1=y;}
  if(x1<0)return {x:out,empty:true};
  x1++;y1++;const side=Math.max(x1-x0,y1-y0,1),s=20/side,cx=(x0+x1)/2,cy=(y0+y1)/2;
  for(let y=0;y<280;y++)for(let x=0;x<280;x++){const v=d[(y*280+x)*4]/255;if(v<=.001)continue;
    const dx=Math.floor((x+.5-cx)*s+14),dy=Math.floor((y+.5-cy)*s+14);if(dx<0||dx>27||dy<0||dy>27)continue;out[dy*28+dx]+=v*s*s;}
  for(let i=0;i<784;i++)out[i]=Math.min(1,out[i]);
  return {x:out,empty:false};}
function padXY(e){const r=pad.getBoundingClientRect();return [(e.clientX-r.left)*280/r.width,(e.clientY-r.top)*280/r.height];}
let padRaf=0;
pad.addEventListener("pointerdown",e=>{if(S.stim){S.stim=null;strokes=[];}pad.setPointerCapture(e.pointerId);cur={w:16,pts:[padXY(e)]};strokes.push(cur);redrawPad();scheduleRun(false);});
pad.addEventListener("pointermove",e=>{if(!cur)return;cur.pts.push(padXY(e));redrawPad();scheduleRun(false);});
const endStroke=()=>{if(!cur)return;cur=null;run(true);};
pad.addEventListener("pointerup",endStroke);pad.addEventListener("pointercancel",endStroke);
function scheduleRun(anim){if(padRaf)return;padRaf=requestAnimationFrame(()=>{padRaf=0;run(anim);});}

/* Example shapes: a port of the generator that made the training drawings */
function preset(cls){S.stim=null;
  const U=(a,b)=>a+Math.random()*(b-a),j=(s=.06)=>U(-s,s);
  let st;
  const seg=(a,b)=>[a,b];
  const sides=c=>{if(Math.random()<.5)return [c];const out=[];for(let i=0;i<c.length-1;i++){const a=c[i],b=c[i+1],d=[b[0]-a[0],b[1]-a[1]],u=U(-.04,.1),v=U(-.04,.1);out.push([[a[0]-d[0]*u,a[1]-d[1]*u],[b[0]+d[0]*v,b[1]+d[1]*v]]);}return out;};
  if(cls===0){const cx=.5+j(.12),cy=.5+j(.12);st=[seg([cx+j(),j()],[cx+j(),1+j()]),seg([j(),cy+j()],[1+j(),cy+j()])];}
  else if(cls===1)st=[seg([j(),j()],[1+j(),1+j()]),seg([1+j(),j()],[j(),1+j()])];
  else if(cls===2){const c=[[j(),j()],[1+j(),j()],[1+j(),1+j()],[j(),1+j()]];c.push(c[0]);st=sides(c);}
  else if(cls===3){const c=[[.5+j(.18),j()],[1+j(),1+j()],[j(),1+j()]];c.push(c[0]);st=sides(c);}
  else{const t0=U(0,6.283),ext=U(-.25,.35),c=[];for(let i=0;i<48;i++){const t=t0+(6.283+ext)*i/47;c.push([.5+.5*Math.cos(t),.5+.5*Math.sin(t)]);}st=[c];}
  const size=U(150,225),asp=cls>=2?U(.75,1.33):U(.85,1.18),sx=size*Math.sqrt(asp),sy=size/Math.sqrt(asp);
  const ang=U(-18,18)*Math.PI/180,ca=Math.cos(ang),sa=Math.sin(ang),w=Math.max(Math.round(U(.055,.12)*size),6),amp=size*U(.004,.03);
  strokes=st.map(pts=>{
    let p=[];for(let i=0;i<pts.length-1;i++)for(let k=0;k<12;k++){const t=k/12;p.push([pts[i][0]+(pts[i+1][0]-pts[i][0])*t,pts[i][1]+(pts[i+1][1]-pts[i][1])*t]);}
    p.push(pts[pts.length-1]);
    const ph=[0,1,2,3].map(()=>U(0,6.283)),n=p.length;
    return {w,pts:p.map((q,i)=>{const t=n>1?i/(n-1):0;
      const ox=Math.sin(6.283*t*1.3+ph[0])+.5*Math.sin(6.283*t*3.1+ph[1]),oy=Math.sin(6.283*t*1.7+ph[2])+.5*Math.sin(6.283*t*2.9+ph[3]);
      const x=(q[0]-.5)*sx+ox*amp,y=(q[1]-.5)*sy+oy*amp;return [x*ca-y*sa+140,x*sa+y*ca+140];})};});
  redrawPad();run(true);}

/* ================= State ================= */
let NET=null,EMPTY=true;
const S={stim:null,lyTouched:{},rfCenter:null,stage:"draw",about:false,sel:null,ly:{1:0,2:1,3:7},rfG:3,decCls:null,
  cv:{L:1,f:5,pos:0,revealed:1e9,sweeping:false,inMap:0},follow:false};
const LG={1:1,2:3,3:5};

/* ================= Colors ================= */
const C={},CS={};
const cssVar=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function readColors(){["bg","surface","fg","muted","grid","accent","ink","neg","const"].forEach(k=>{CS[k]=cssVar("--"+k)||"#888";if(typeof THREE!=="undefined")C[k]=new THREE.Color(CS[k]);});}
readColors();
const hexA=(hex,a)=>{const c=hex.replace("#","");const n=parseInt(c.length===3?c.split("").map(x=>x+x).join(""):c,16);return `rgba(${n>>16&255},${n>>8&255},${n&255},${a})`;};

/* ================= 3D scene ================= */
let renderer,scene,camera,controls,GL=false;
const stageEl=$("#stage"),canvas=$("#gl");
const meshes=[],planes=[],frames=[],labels=[],dyn=[];
let avgMesh,outMesh,orb,winBox,winLines,rfGroup,selBox;
const HAS3=typeof THREE!=="undefined";
const dummy=HAS3?new THREE.Object3D():null;
function initGL(){
  try{renderer=new THREE.WebGLRenderer({canvas,antialias:true});}catch(e){const d=document.createElement("div");d.className="glerr";d.textContent="This view needs WebGL, which isn't available here. The panel still works.";stageEl.appendChild(d);return;}
  GL=true;renderer.setPixelRatio(Math.min(devicePixelRatio,2));
  scene=new THREE.Scene();camera=new THREE.PerspectiveCamera(38,1,.1,500);
  controls=new THREE.OrbitControls(camera,canvas);controls.enableDamping=true;controls.dampingFactor=.09;controls.minDistance=3;controls.maxDistance=120;
  const unit=new THREE.BoxGeometry(1,1,1);
  GR.forEach((g,gi)=>{
    if(g.H>1){
      const m=new THREE.InstancedMesh(unit,new THREE.MeshBasicMaterial(),g.n*g.H*g.H);
      m.instanceMatrix.setUsage(THREE.DynamicDrawUsage);for(let i=0;i<m.count;i++){m.setColorAt(i,new THREE.Color());}
      scene.add(m);meshes[gi]=m;
      for(let ch=0;ch<g.n;ch++){
        const x=g.x0+ch*g.gap;
        const pl=new THREE.Mesh(new THREE.PlaneGeometry(SZ,SZ),new THREE.MeshBasicMaterial({transparent:true,opacity:.3,side:THREE.DoubleSide,depthWrite:false}));
        pl.rotation.y=Math.PI/2;pl.position.set(x,0,0);pl.userData={gi,ch};scene.add(pl);planes.push(pl);
        const h=SZ/2,fr=new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints([[x,h,-h],[x,h,h],[x,-h,h],[x,-h,-h]].map(p=>new THREE.Vector3(...p))),new THREE.LineBasicMaterial({transparent:true,opacity:.5}));
        fr.userData={gi,ch};scene.add(fr);frames.push(fr);}
    }
  });
  avgMesh=new THREE.InstancedMesh(unit,new THREE.MeshBasicMaterial(),16);for(let i=0;i<16;i++)avgMesh.setColorAt(i,new THREE.Color());scene.add(avgMesh);meshes[6]=avgMesh;
  outMesh=new THREE.InstancedMesh(unit,new THREE.MeshBasicMaterial(),5);for(let i=0;i<5;i++)outMesh.setColorAt(i,new THREE.Color());scene.add(outMesh);meshes[7]=outMesh;
  // pick targets for avg/out
  for(let k=0;k<16;k++){const p=new THREE.Mesh(new THREE.BoxGeometry(.4,.34,.4),new THREE.MeshBasicMaterial({visible:false}));p.position.set(GR[6].x0,avgY(k),0);p.userData={gi:6,ch:k};scene.add(p);planes.push(p);}
  // labels
  GR.forEach((g,gi)=>{
    labels.push(makeLabel(g.label,[g.xm,g.lblY||SZ/2+.7,0],.02,"fg"));
    const rf=gi===0?"1 pixel each":gi>=6?"sees everything":`sees ${g.rf}×${g.rf} px`;
    if(gi<7)labels.push(makeLabel(rf,[g.xm,g.lblY?-3.8:-SZ/2-.6,0],.016,gi>=6?"accent":"ink"));});
  for(let k=0;k<5;k++)dyn.push(makeLabel("",[GR[7].x0+4.8,outY(k),0],.018,"fg"));
  // connectors
  const conn=[[GR[0].x1,GR[1].x0],[GR[1].x1,GR[2].x0],[GR[2].x1,GR[3].x0],[GR[3].x1,GR[4].x0],[GR[4].x1,GR[5].x0],[GR[5].x1,GR[6].x0],[GR[6].x0,GR[7].x0]];
  conn.forEach(([a,b])=>{const l=new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(a+.15,0,0),new THREE.Vector3(b-.15,0,0)]),new THREE.LineBasicMaterial({transparent:true,opacity:.6}));l.userData.conn=1;scene.add(l);frames.push(l);});
  // conv window + lines
  winBox=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1,1,1)),new THREE.LineBasicMaterial({linewidth:2}));scene.add(winBox);
  winLines=new THREE.LineSegments(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({transparent:true,opacity:.85}));
  winLines.geometry.setAttribute("position",new THREE.Float32BufferAttribute(new Float32Array(8*3),3));scene.add(winLines);
  selBox=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1,1,1)),new THREE.LineBasicMaterial());scene.add(selBox);
  rfGroup=new THREE.Group();scene.add(rfGroup);
  orb=new THREE.Sprite(new THREE.SpriteMaterial({map:glowTex(),transparent:true,depthTest:false}));orb.scale.set(2.4,2.4,1);orb.renderOrder=30;orb.visible=false;scene.add(orb);
  applyColors3D();
  canvas.addEventListener("pointerdown",e=>{downAt=[e.clientX,e.clientY];});
  canvas.addEventListener("pointerup",e=>{if(!downAt)return;const d=Math.hypot(e.clientX-downAt[0],e.clientY-downAt[1]);downAt=null;if(d<6)pick(e);});
  new ResizeObserver(resize).observe(stageEl);resize();
  camera.position.set(...VIEWS.draw.p);controls.target.set(...VIEWS.draw.t);
}
const avgY=k=>3.0-k*.4, outY=k=>2.4-k*1.2;
let downAt=null;
function glowTex(){const cv=document.createElement("canvas");cv.width=cv.height=128;const x=cv.getContext("2d");const g=x.createRadialGradient(64,64,0,64,64,64);
  g.addColorStop(0,hexA(CS.accent,1));g.addColorStop(.3,hexA(CS.accent,.55));g.addColorStop(1,hexA(CS.accent,0));x.fillStyle=g;x.fillRect(0,0,128,128);return new THREE.CanvasTexture(cv);}
function makeLabel(text,pos,size,col){
  const sp=new THREE.Sprite(new THREE.SpriteMaterial({transparent:true,depthTest:false,sizeAttenuation:false}));sp.renderOrder=20;
  sp.position.set(...pos);sp.userData={text,size,col};scene.add(sp);drawLabel(sp);return sp;}
function drawLabel(sp){const {text,size,col}=sp.userData;const cv=document.createElement("canvas");const x=cv.getContext("2d");
  const f=`600 44px "Familjen Grotesk","Helvetica Neue",Arial,sans-serif`;x.font=f;const w=Math.max(8,Math.ceil(x.measureText(text).width)+16);
  cv.width=w;cv.height=60;x.font=f;x.fillStyle=CS[col]||CS.fg;x.textBaseline="middle";x.textAlign="center";x.fillText(text,w/2,32);
  if(sp.material.map)sp.material.map.dispose();sp.material.map=new THREE.CanvasTexture(cv);sp.material.map.anisotropy=4;sp.material.needsUpdate=true;sp.userData.bx=size*w/60;sp.scale.set(size*w/60*LF,size*LF,1);}
let LF=1;
function setLabel(sp,text,col){if(sp.userData.text===text&&sp.userData.col===col)return;sp.userData.text=text;if(col)sp.userData.col=col;drawLabel(sp);}
function applyColors3D(){if(!GL)return;renderer.setClearColor(C.bg);
  planes.forEach(p=>{if(p.material.visible!==false&&p.material.color)p.material.color.copy(C.surface);});
  frames.forEach(f=>f.material.color.copy(C.grid));
  winBox.material.color.copy(C.ink);winLines.material.color.copy(C.ink);selBox.material.color.copy(C.ink);
  labels.concat(dyn).forEach(drawLabel);orb.material.map=glowTex();orb.material.needsUpdate=true;}
function resize(){if(!GL)return;const w=stageEl.clientWidth,h=stageEl.clientHeight;if(!w||!h)return;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();}

/* ================= Camera views ================= */
const VIEWS={
  draw:{p:[13,19,80],t:[26.5,-1.5,0]},
  conv1:{p:[-8.5,5,9.5],t:[3.5,-.4,0]},
  conv2:{p:[6.2,5.5,11],t:[16.5,-.4,0]},
  l1:{p:[-5,6,12],t:[5.5,-.5,0]},
  l2:{p:[9.5,6,13],t:[20,-.5,0]},
  l3:{p:[25,6,14],t:[36.5,-.5,0]},
  rf:{p:[4,10,34],t:[20,-1,0]},
  dec:{p:[27.5,8,31],t:[39.5,-.3,0]}};
let fly=null;
function flyTo(v){if(!GL)return;const P=new THREE.Vector3(...v.p),T=new THREE.Vector3(...v.t);
  if(reduced){camera.position.copy(P);controls.target.copy(T);return;}
  fly={p0:camera.position.clone(),t0:controls.target.clone(),P,T,start:performance.now(),dur:900};}
function viewFor(){const s=S.stage;if(s==="conv")return S.cv.L===1?VIEWS.conv1:VIEWS.conv2;
  if(s==="rf"){const xe=GR[S.rfG].xm,tx=xe/2,sp=xe+SZ;return {t:[tx+.5,-.6,0],p:[tx-sp*.42,6+sp*.16,11+sp*.78]};}
  return VIEWS[s]||VIEWS.draw;}

/* ================= Signal wave ================= */
let wave=null;
function startWave(){if(reduced||!GL){wave=null;return;}wave={start:performance.now(),dur:1500};}
function gProg(gi){if(!wave)return 1;const t=(performance.now()-wave.start)/wave.dur;return clamp(t*(NG+.5)-gi,0,1);}
const easeBack=k=>{const c=1.70158;return 1+(c+1)*Math.pow(k-1,3)+c*Math.pow(k-1,2);};

/* ================= Cell updates ================= */
function layerMax(arr){let m=1e-6;for(let i=0;i<arr.length;i++)if(arr[i]>m)m=arr[i];return m;}
function updateCells(){
  if(!GL||!NET)return;
  const st=S.stage,tmp=new THREE.Color();
  const focusG=st==="l1"||st==="l2"||st==="l3"?LG[st[1]]:-1, focusCh=focusG>0?S.ly[st[1]]:-1;
  const cv=S.cv,convIn=cv.L===1?0:2,convOut=cv.L===1?1:3;
  for(let gi=0;gi<6;gi++){
    const g=GR[gi],m=meshes[gi],A=groupData(gi),HH=g.H*g.H,mx=gi===0?1:layerMax(A)*.85;
    const pk=easeBack(gProg(gi));let fadeG=0;
    if(focusG>=0&&gi!==focusG)fadeG=.55;
    if(st==="conv"&&gi!==convIn&&gi!==convOut)fadeG=.7;
    if(st==="dec"&&gi>0)fadeG=.45;
    for(let ch=0;ch<g.n;ch++){
      let fade=fadeG;if(gi===focusG&&ch!==focusCh)fade=.72;
      let hideFrom=1e9,hide=false,shrink=1;
      if(st==="conv"&&gi===convOut){if(ch!==cv.f)hide=true;else hideFrom=cv.revealed;}
      if(gi===focusG&&ch!==focusCh){shrink=.5;fade=.8;}
      for(let i=0;i<HH;i++){
        const idx=ch*HH+i,r=(i/g.H)|0,c=i%g.H;let v=A[ch*HH+i]/mx;v=Math.min(1,v);
        if(i>=hideFrom)v=0;
        const x=g.x0+ch*g.gap,y=SZ/2-(r+.5)*g.cs,z=(c+.5)*g.cs-SZ/2;
        if(v<.1||hide){dummy.scale.set(0,0,0);}else{
          const s=(.38+.62*v)*g.cs*.9*pk*shrink;dummy.scale.set(g.th*(.4+.6*v)*Math.max(pk,.001)+.001,s,s);}
        dummy.position.set(x,y,z);dummy.updateMatrix();m.setMatrixAt(idx,dummy.matrix);
        if(gi===0){
          tmp.copy(C.bg).lerp(C.ink,.35+.65*v);
        }else tmp.copy(C.bg).lerp(C.accent,.28+.72*v);
        if(fade)tmp.lerp(C.bg,fade);
        m.setColorAt(idx,tmp);}
    }
    m.instanceMatrix.needsUpdate=true;m.instanceColor.needsUpdate=true;
  }
  // average column
  const gmx=layerMax(NET.g),pk6=easeBack(gProg(6));
  const dcls=S.decCls??NET.win;
  for(let k=0;k<16;k++){const v=NET.g[k]/gmx;const s=(.08+.3*v)*Math.max(pk6,.001);
    dummy.position.set(GR[6].x0,avgY(k),0);dummy.scale.set(s,s,s);dummy.updateMatrix();avgMesh.setMatrixAt(k,dummy.matrix);
    if(st==="dec"){const ct=WD[dcls*16+k]*NET.g[k];tmp.copy(C.bg).lerp(ct>=0?C.accent:C.neg,clamp(.25+Math.abs(ct)*.8,0,1));}
    else tmp.copy(C.bg).lerp(C.accent,.3+.7*v);
    avgMesh.setColorAt(k,tmp);}
  avgMesh.instanceMatrix.needsUpdate=true;avgMesh.instanceColor.needsUpdate=true;
  const pk7=easeBack(gProg(7));
  for(let k=0;k<5;k++){const p=EMPTY?0:NET.prob[k];const L=(.12+4*p)*Math.max(pk7,.001);
    dummy.position.set(GR[7].x0+L/2,outY(k),0);dummy.scale.set(L,.62,.62);dummy.updateMatrix();outMesh.setMatrixAt(k,dummy.matrix);
    const hi=st==="dec"?k===dcls:k===NET.win&&!EMPTY;tmp.copy(C.bg).lerp(hi?C.accent:C.muted,hi?1:.45);outMesh.setColorAt(k,tmp);
    setLabel(dyn[k],`${GLY[k]}  ${CLS[k]}  ${EMPTY?"–":pct(NET.prob[k])}`,hi?"accent":"fg");
    dyn[k].position.x=GR[7].x0+4.6;dyn[k].center.set(0,.5);}
  outMesh.instanceMatrix.needsUpdate=true;outMesh.instanceColor.needsUpdate=true;
  // frames highlight
  frames.forEach(f=>{if(f.userData.conn)return;const {gi,ch}=f.userData;
    const on=(gi===focusG&&ch===focusCh)||(st==="conv"&&gi===convOut&&ch===cv.f);
    f.material.color.copy(on?C.accent:C.grid);f.material.opacity=on?1:.5;});
  const anyFocus=focusG>=0||st==="conv";
  planes.forEach(p=>{if(p.userData.gi>=6)return;const {gi,ch}=p.userData;
    const on=(gi===focusG&&ch===focusCh)||(st==="conv"&&((gi===convOut&&ch===cv.f)||gi===convIn));
    p.material.opacity=anyFocus?(on?.55:.07):.3;});
}

/* ================= Overlays: conv window, receptive-field cone ================= */
function setBox(obj,x0,x1,y0,y1,z0,z1){obj.position.set((x0+x1)/2,(y0+y1)/2,(z0+z1)/2);obj.scale.set(Math.abs(x1-x0),Math.abs(y1-y0),Math.abs(z1-z0));}
function updateOverlays(){
  if(!GL)return;
  const st=S.stage;
  // conv window
  if(st==="conv"&&NET){const cv=S.cv,gi=cv.L===1?0:2,go=cv.L===1?1:3,g=GR[gi],H=g.H,r=(cv.pos/H)|0,c=cv.pos%H;
    const y0=SZ/2-(r-1)*g.cs,y1=SZ/2-(r+2)*g.cs,z0=(c-1)*g.cs-SZ/2,z1=(c+2)*g.cs-SZ/2;
    const xa=g.x0-.22,xb=g.x1+.22;setBox(winBox,xa,xb,y0,y1,z0,z1);winBox.visible=true;
    const [ox,oy,oz]=cellPos(go,cv.f,r,c),h=GR[go].cs/2,P=winLines.geometry.attributes.position.array;
    const cs=[[y0,z0],[y0,z1],[y1,z0],[y1,z1]],oc=[[oy+h,oz-h],[oy+h,oz+h],[oy-h,oz-h],[oy-h,oz+h]];
    for(let k=0;k<4;k++){P.set([xb,cs[k][0],cs[k][1],ox,oc[k][0],oc[k][1]],k*6);}
    winLines.geometry.attributes.position.needsUpdate=true;winLines.geometry.computeBoundingSphere();winLines.visible=true;
    setBox(selBox,ox-.2,ox+.2,oy-h,oy+h,oz-h,oz+h);selBox.visible=true;
  }else{winBox.visible=false;winLines.visible=false;selBox.visible=false;}
  // RF cone
  while(rfGroup.children.length){const o=rfGroup.children.pop();o.geometry.dispose();o.material.dispose();}
  if(st==="conv"||!S.sel)return;
  const {gi,ch,r,c}=S.sel;
  let top;// 4 corners of the selected thing
  if(gi===6){const y=avgY(ch),x=GR[6].x0;top=[[x,y+.18,-.18],[x,y+.18,.18],[x,y-.18,-.18],[x,y-.18,.18]];setBox(selBox,x-.25,x+.25,y-.25,y+.25,-.25,.25);}
  else{const [x,y,z]=cellPos(gi,ch,r,c),h=GR[gi].cs/2;top=[[x,y+h,z-h],[x,y+h,z+h],[x,y-h,z-h],[x,y-h,z+h]];setBox(selBox,x-.18,x+.18,y-h,y+h,z-h,z+h);}
  selBox.visible=true;
  if(st==="dec"&&gi===5){const [x,y,z]=cellPos(5,ch,r,c),v=WD[(S.decCls??NET.win)*16+ch]*NET.g[ch];
    rfGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(x,y,z),new THREE.Vector3(GR[6].x0,avgY(ch),0)]),new THREE.LineBasicMaterial({color:v>=0?C.accent:C.neg})));}
  const R=gi===6?{0:[0,27,0,27],1:[0,27,0,27],2:[0,13,0,13],3:[0,13,0,13],4:[0,6,0,6],5:[0,6,0,6]}:rfRanges(gi,r,c);
  const mat=()=>new THREE.LineBasicMaterial({color:C.ink,transparent:true,opacity:.95});
  const lastGi=gi===6?5:gi-1;
  for(let k=0;k<=lastGi;k++){const g=GR[k];let [a,b,e,f]=R[k];a=clamp(a,0,g.H-1);b=clamp(b,0,g.H-1);e=clamp(e,0,g.H-1);f=clamp(f,0,g.H-1);
    const x=k===0?g.x0-.16:g.x1+.02,y0=SZ/2-a*g.cs,y1=SZ/2-(b+1)*g.cs,z0=e*g.cs-SZ/2,z1=(f+1)*g.cs-SZ/2;
    const pts=[[x,y0,z0],[x,y0,z1],[x,y1,z1],[x,y1,z0]].map(p=>new THREE.Vector3(...p));
    const loop=new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(pts),mat());rfGroup.add(loop);
    if(k===0){ // shaded footprint on the input + cone lines
      const fill=new THREE.Mesh(new THREE.PlaneGeometry(Math.abs(z1-z0),Math.abs(y1-y0)),new THREE.MeshBasicMaterial({color:C.ink,transparent:true,opacity:.16,side:THREE.DoubleSide,depthWrite:false}));
      fill.rotation.y=Math.PI/2;fill.position.set(x-.01,(y0+y1)/2,(z0+z1)/2);rfGroup.add(fill);
      const cn=[[x,y0,z0],[x,y0,z1],[x,y1,z0],[x,y1,z1]],arr=[];
      for(let q=0;q<4;q++)arr.push(new THREE.Vector3(...cn[q]),new THREE.Vector3(...top[q]));
      const m2=mat();m2.opacity=.55;rfGroup.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(arr),m2));}
  }
}

/* ================= Picking ================= */
const ray=HAS3?new THREE.Raycaster():null,mouse=HAS3?new THREE.Vector2():null;
function pick(e){
  const r=canvas.getBoundingClientRect();mouse.set(((e.clientX-r.left)/r.width)*2-1,-((e.clientY-r.top)/r.height)*2+1);
  ray.setFromCamera(mouse,camera);const hit=ray.intersectObjects(planes,false)[0];if(!hit)return;
  const {gi,ch}=hit.object.userData;
  if(gi===6){S.sel={gi:6,ch,r:0,c:0};if(S.stage==="draw"||S.stage==="rf"){S.rfG=6;}refresh();return;}
  const g=GR[gi],p=hit.point,c=clamp(Math.floor((p.z+SZ/2)/g.cs),0,g.H-1),rr=clamp(Math.floor((SZ/2-p.y)/g.cs),0,g.H-1);
  if(S.stage==="conv"){const cv=S.cv,gin=cv.L===1?0:2,gout=cv.L===1?1:3;
    if(gi===gin||gi===gout){stopSweep();cv.pos=rr*g.H+c;if(gi===gout)cv.f=ch;refresh();}
    return;}
  if(gi===0){S.sel=null;refresh();return;}
  S.sel={gi,ch,r:rr,c};
  if(S.stage.startsWith("l")){const L=+S.stage[1];if(LG[L]===gi){S.ly[L]=ch;}}
  if(S.stage==="rf"||S.stage==="draw")S.rfG=gi;
  refresh();
}

/* ================= Panel: shared helpers ================= */
function barsHTML(vals,win,opts={}){return vals.map((v,k)=>`<div class="b${k===win?" win":""}${opts.sel===k?" sel":""}" data-k="${k}"><span class="g">${GLY[k]}</span><span>${CLS[k]}</span><span class="t"><i style="width:${(v*100).toFixed(1)}%"></i></span><em>${opts.fmt?opts.fmt(v):pct(v)}</em></div>`).join("");}
function heatCell(v,max,pos,neg){const a=clamp(Math.abs(v)/max,0,1);return `background:${hexA(v>=0?pos:neg,.12+.78*a)};color:${a>.55?CS.surface:CS.fg}`;}
function drawMap(cv,data,H,opts={}){ // feature map over faint input
  const x=cv.getContext("2d"),W=cv.width,s=W/H;x.imageSmoothingEnabled=false;x.fillStyle=CS.bg;x.fillRect(0,0,W,W);
  if(opts.under&&NET){const t=W/28;for(let i=0;i<784;i++){const v=NET.x[i];if(v>.02){x.fillStyle=hexA(CS.fg,.16*v);x.fillRect((i%28)*t,((i/28)|0)*t,t+.5,t+.5);}}}
  const mx=opts.max||layerMax(data);
  for(let i=0;i<H*H;i++){if(opts.limit!=null&&i>=opts.limit)break;const v=data[i]/mx;if(v<=.1)continue;x.fillStyle=hexA(opts.col||CS.accent,clamp(.15+.85*v,0,1)*(opts.alpha||1));x.fillRect((i%H)*s,((i/H)|0)*s,s+.5,s+.5);}
  if(opts.mark){const [r,c,n]=opts.mark;x.strokeStyle=CS.ink;x.lineWidth=2;x.strokeRect(c*s+1,r*s+1,(n||1)*s-2,(n||1)*s-2);}
}

/* ================= Pixel view (28x28) ================= */
const px=$("#px"),pxx=px.getContext("2d");
function drawPixelView(){
  const st=S.stage,t=10;pxx.fillStyle=CS.bg;pxx.fillRect(0,0,280,280);
  const ink=(alpha)=>{for(let i=0;i<784;i++){const v=NET.x[i];if(v>.01){pxx.fillStyle=hexA(CS.ink,clamp(v,0,1)*alpha);pxx.fillRect((i%28)*t,((i/28)|0)*t,t,t);}}};
  const gridl=(n,op)=>{pxx.strokeStyle=hexA(CS.grid,op);pxx.lineWidth=1;pxx.beginPath();const s=280/n;for(let k=1;k<n;k++){pxx.moveTo(k*s+.5,0);pxx.lineTo(k*s+.5,280);pxx.moveTo(0,k*s+.5);pxx.lineTo(280,k*s+.5);}pxx.stroke();};
  const box=(r0,c0,n,col=CS.ink,w=3)=>{pxx.strokeStyle=col;pxx.lineWidth=w;pxx.strokeRect(c0*t+w/2,r0*t+w/2,n*t-w,n*t-w);};
  let cap="";
  if(st==="conv"){const cv=S.cv;
    if(cv.L===1){ink(1);gridl(28,.5);const r=(cv.pos/28)|0,c=cv.pos%28;box(r-1,c-1,3);cap=`Input pixels · window centred on row ${r}, column ${c}`;}
    else{const k=cv.inMap,s=20,A=NET.p1,mx=layerMax(A);for(let i=0;i<196;i++){const v=A[k*196+i]/mx;if(v>.1){pxx.fillStyle=hexA(CS.accent,clamp(.15+.85*v,0,1));pxx.fillRect((i%14)*s,((i/14)|0)*s,s,s);}}
      gridl(14,.5);const r=(cv.pos/14)|0,c=cv.pos%14;pxx.strokeStyle=CS.ink;pxx.lineWidth=3;pxx.strokeRect((c-1)*s+1.5,(r-1)*s+1.5,3*s-3,3*s-3);
      cap=`Pooled layer-1 map ${k} (“${M.layers[0].channels[k].name}”) · 14×14`;}
  }else if(st==="l1"||st==="l2"||st==="l3"){const L=+st[1],gi=LG[L],g=GR[gi],ch=S.ly[L],A=groupData(gi).subarray(ch*g.H*g.H,(ch+1)*g.H*g.H),mx=layerMax(A),s=280/g.H;
    ink(.35);for(let i=0;i<g.H*g.H;i++){const v=A[i]/mx;if(v>.1){pxx.fillStyle=hexA(CS.accent,clamp(.1+.7*v,0,.8));pxx.fillRect((i%g.H)*s,((i/g.H)|0)*s,s,s);}}
    if(S.sel&&S.sel.gi===gi){const R=rfRanges(gi,S.sel.r,S.sel.c)[0];box(R[0],R[2],R[1]-R[0]+1);
      pxx.strokeStyle=CS.ink;pxx.lineWidth=2;pxx.strokeRect(S.sel.c*s+1,S.sel.r*s+1,s-2,s-2);
      cap=`Teal: where “${M.layers[L-1].channels[ch].name}” fires · amber: what the selected cell sees (${g.rf}×${g.rf} px)`;}
    else cap=`Teal: where “${M.layers[L-1].channels[ch].name}” fires on your drawing · click to see a cell’s view`;
  }else if(st==="rf"){ink(1);gridl(28,.35);
    const sel=rfSel(),R=sel.R;if(R){const a=Math.max(0,R[0]),b=Math.min(27,R[1]),e=Math.max(0,R[2]),f=Math.min(27,R[3]);
      pxx.fillStyle=hexA(CS.bg,.72);pxx.beginPath();pxx.rect(0,0,280,280);pxx.rect(e*t,a*t,(f-e+1)*t,(b-a+1)*t);pxx.fill("evenodd");box(a,e,0);pxx.strokeStyle=CS.ink;pxx.lineWidth=3;pxx.strokeRect(e*t+1.5,a*t+1.5,(f-e+1)*t-3,(b-a+1)*t-3);}
    cap=`Amber: the pixels one ${GR[S.rfG].key==="avg"?"average":GR[S.rfG].label.split(" · ")[0].toLowerCase()} cell depends on · click to move it`;
  }else if(st==="dec"&&decTop){ink(.75);gridl(28,.25);
    decTop.forEach(o=>{const i=NET.gi[o.k],r=(i/7)|0,c=i%7,R=rfRanges(5,r,c)[0],a=Math.max(0,R[0]),b=Math.min(27,R[1]),e=Math.max(0,R[2]),f=Math.min(27,R[3]);
      const on=o.k===S.decFocus,col=o.v>=0?CS.accent:CS.neg;
      if(on){pxx.fillStyle=hexA(col,.16);pxx.fillRect(e*t,a*t,(f-e+1)*t,(b-a+1)*t);}
      pxx.strokeStyle=on?col:hexA(col,.45);pxx.lineWidth=on?4:2;pxx.setLineDash(on?[]:[6,4]);pxx.strokeRect(e*t+2,a*t+2,(f-e+1)*t-4,(b-a+1)*t-4);pxx.setLineDash([]);
      pxx.fillStyle=on?col:hexA(col,.7);pxx.font="600 15px JetBrains Mono, monospace";pxx.fillText(String(o.k),e*t+6,a*t+18);});
    const fo=M.layers[2].channels[S.decFocus];cap=`Each box: where a layer-3 channel found its best match (18×18 px). Solid: ${S.decFocus} · ${fo.name}`;}
  $("#pxCap").textContent=cap;
}
const hexToRgb=h=>{const n=parseInt(h.replace("#",""),16);return [n>>16&255,n>>8&255,n&255];};
px.addEventListener("click",e=>{const r=px.getBoundingClientRect(),X=(e.clientX-r.left)/r.width*28,Y=(e.clientY-r.top)/r.height*28;
  const st=S.stage;
  if(st==="conv"){stopSweep();const H=S.cv.L===1?28:14,f=H/28;S.cv.pos=clamp(Math.floor(Y*f),0,H-1)*H+clamp(Math.floor(X*f),0,H-1);refresh();return;}
  if(st.startsWith("l")){const L=+st[1],gi=LG[L],H=GR[gi].H;S.sel={gi,ch:S.ly[L],r:clamp(Math.floor(Y*H/28),0,H-1),c:clamp(Math.floor(X*H/28),0,H-1)};refresh();return;}
  if(st==="rf"){S.rfCenter=[Y,X];S.sel=null;refresh();}
});
function rfSel(){ // receptive field for the rf stage
  const gi=S.rfG;if(gi>=6)return {R:[0,27,0,27],gi};
  if(S.sel&&S.sel.gi===gi)return {R:rfRanges(gi,S.sel.r,S.sel.c)[0],gi,r:S.sel.r,c:S.sel.c};
  const [Y,X]=S.rfCenter||[14,14],H=GR[gi].H,r=clamp(Math.floor(Y*H/28),0,H-1),c=clamp(Math.floor(X*H/28),0,H-1);
  return {R:rfRanges(gi,r,c)[0],gi,r,c};}

/* ================= Draw panel ================= */
function refreshDraw(){
  const st=$("#status"),pb=$("#predBars");
  if(EMPTY){st.innerHTML=`Draw a shape<small>The network knows five: plus, cross, square, triangle, circle</small>`;pb.innerHTML=barsHTML([0,0,0,0,0],-1,{fmt:()=>"–"});}
  else{const w=NET.win,p=NET.prob[w];st.innerHTML=`${GLY[w]} ${CLS[w]} · ${pct(p)}<small>${p>.9?"Confident":p>.6?"Fairly sure":"Unsure, hedging between shapes"} · click any cell in 3D to see what it can see</small>`;pb.innerHTML=barsHTML(NET.prob,w);}
  const sc=$("#seen").getContext("2d"),im=sc.createImageData(28,28),bg=hexToRgb(CS.bg),fg=hexToRgb(CS.ink);
  for(let i=0;i<784;i++){const v=NET?NET.x[i]:0;im.data.set([bg[0]+(fg[0]-bg[0])*v,bg[1]+(fg[1]-bg[1])*v,bg[2]+(fg[2]-bg[2])*v,255],i*4);}
  sc.putImageData(im,0,0);
}

/* ================= Convolution panel ================= */
let sweepTimer=null;
function convCalc(){const cv=S.cv,L=cv.L,H=L===1?28:14,C=L===1?1:8,X=L===1?NET.x:NET.p1,W=L===1?W1:W2,B=L===1?B1:B2,r=(cv.pos/H)|0,c=cv.pos%H;
  const per=[];let tot=0;
  for(let k=0;k<C;k++){const patch=[],ker=[],prod=[];let s=0;
    for(let dr=-1;dr<=1;dr++)for(let dc=-1;dc<=1;dc++){const rr=r+dr,cc=c+dc,v=rr<0||rr>=H||cc<0||cc>=H?0:X[k*H*H+rr*H+cc],w=W[(cv.f*C+k)*9+(dr+1)*3+dc+1];patch.push(v);ker.push(w);prod.push(v*w);s+=v*w;}
    per.push({patch,ker,prod,s});tot+=s;}
  return {H,C,r,c,per,tot,b:B[cv.f],z:tot+B[cv.f]};}
function kernelThumb(cvs,W,off,C=1){const x=cvs.getContext("2d");cvs.width=cvs.height=3;const im=x.createImageData(3,3);let mx=1e-6;for(let i=0;i<9;i++)mx=Math.max(mx,Math.abs(W[off+i]));
  const pc=hexToRgb(CS.accent),nc=hexToRgb(CS.neg),bg=hexToRgb(CS.bg);
  for(let i=0;i<9;i++){const v=W[off+i]/mx,c=v>=0?pc:nc,a=Math.abs(v);im.data.set([bg[0]+(c[0]-bg[0])*a,bg[1]+(c[1]-bg[1])*a,bg[2]+(c[2]-bg[2])*a,255],i*4);}x.putImageData(im,0,0);}
function refreshConv(){
  const cv=S.cv,L=cv.L,F=L===1?8:12,names=M.layers[L-1].channels;
  $("#cv1").setAttribute("aria-pressed",L===1);$("#cv2").setAttribute("aria-pressed",L===2);
  $("#cvInWrap").hidden=L===1;$("#cv2note").hidden=L===1;
  const chips=$("#cvChips");
  if(chips.dataset.L!==String(L)||chips.dataset.theme!==CS.bg){chips.dataset.L=L;chips.dataset.theme=CS.bg;chips.innerHTML="";
    for(let f=0;f<F;f++){const b=document.createElement("button");b.setAttribute("aria-label",`Filter ${f}: ${names[f].name}`);b.title=names[f].name;const c=document.createElement("canvas");b.appendChild(c);
      b.onclick=()=>{stopSweep();cv.f=f;cv.revealed=1e9;refresh();};chips.appendChild(b);}}
  [...chips.children].forEach((b,f)=>{b.setAttribute("aria-pressed",f===cv.f);const c=b.firstChild;
    if(L===1)kernelThumb(c,W1,f*9);else{c.width=c.height=14;drawMap(c,NET.a2.subarray(f*196,(f+1)*196),14);}});
  $("#cvFiltName").textContent=`${cv.f} · ${names[cv.f].name}`;
  const k=L===1?0:cv.inMap;$("#cvIn").value=k;$("#cvInName").textContent=`${k} · ${M.layers[0].channels[k].name}`;
  const R=convCalc(),P=R.per[k];
  const mxW=Math.max(...P.ker.map(Math.abs),1e-6),mxP=Math.max(...P.prod.map(Math.abs),1e-6);
  const grid=(arr,sty,d)=>`<div class="g">${arr.map(v=>`<span style="${sty(v)}">${fmt(v,d)}</span>`).join("")}</div>`;
  const inCol=L===1?CS.ink:CS.accent,mxIn=L===1?1:layerMax(NET.p1);
  $("#cvGrids").innerHTML=`<div class="kg">${L===1?"pixels":"map "+k}${grid(P.patch,v=>heatCell(v,mxIn,inCol,inCol),2)}</div><span class="op">×</span>
    <div class="kg">weights${grid(P.ker,v=>heatCell(v,mxW,CS.accent,CS.neg),2)}</div><span class="op">=</span>
    <div class="kg">products${grid(P.prod,v=>heatCell(v,mxP,CS.accent,CS.neg),2)}</div>`;
  const a=Math.max(0,R.z);
  let html=`<span class="k">position</span> row ${R.r}, col ${R.c}<br>`;
  if(L===1)html+=`<span class="k">Σ 9 products</span> ${fmt(R.tot,3)}<br>`;
  else{html+=R.per.map((p,i)=>`<span class="k">map ${i}</span> ${p.s>=0?"+":""}${fmt(p.s,2)}`).join(" &nbsp;")+`<br><span class="k">Σ 72 products</span> ${fmt(R.tot,3)}<br>`;}
  html+=`<span class="k">+ bias</span> ${fmt(R.b,3)} <span class="k">=</span> ${fmt(R.z,3)}<br><span class="k">ReLU →</span> <span class="${a>0?"hit":"k"}">${fmt(a,3)}</span>${a>0?"":" (negative, so zero)"}`;
  $("#cvReadout").innerHTML=html;
  const H=R.H,data=(L===1?NET.a1:NET.a2).subarray(cv.f*H*H,(cv.f+1)*H*H);
  drawMap($("#cvMini"),data,H,{limit:Math.min(cv.revealed,H*H),mark:[R.r,R.c,1],max:layerMax(data)});
  $("#sweepBtn").textContent=cv.sweeping?"❚❚ Pause":"▶ Sweep";
}
function stopSweep(){S.cv.sweeping=false;clearInterval(sweepTimer);sweepTimer=null;if(S.cv.revealed<1e9&&S.cv.revealed<=S.cv.pos)S.cv.revealed=S.cv.pos+1;}
function startSweep(){const cv=S.cv,H=cv.L===1?28:14;if(cv.revealed>=H*H||cv.revealed>=1e9){cv.pos=0;cv.revealed=1;}
  cv.sweeping=true;clearInterval(sweepTimer);
  sweepTimer=setInterval(()=>{const n=$("#fast").checked?(cv.L===1?14:4):(cv.L===1?2:1);
    for(let i=0;i<n;i++){if(cv.pos>=H*H-1){cv.revealed=H*H;stopSweep();cv.revealed=1e9;break;}cv.pos++;cv.revealed=cv.pos+1;}refresh();},cv.L===1?40:70);
  refresh();}
$("#sweepBtn").onclick=()=>{if(S.cv.sweeping)stopSweep();else startSweep();refresh();};
$("#stepBtn").onclick=()=>{stopSweep();const H=S.cv.L===1?28:14;S.cv.pos=(S.cv.pos+1)%(H*H);if(S.cv.revealed<1e9)S.cv.revealed=Math.max(S.cv.revealed,S.cv.pos+1);refresh();};
function convToPeak(){const cv=S.cv,H=cv.L===1?28:14,A=(cv.L===1?NET.a1:NET.a2).subarray(cv.f*H*H,(cv.f+1)*H*H);let m=-1,mi=0;for(let i=0;i<A.length;i++)if(A[i]>m){m=A[i];mi=i;}cv.pos=mi;}
$("#cv1").onclick=()=>{stopSweep();Object.assign(S.cv,{L:1,f:5,revealed:1e9});convToPeak();refresh();flyTo(viewFor());};
$("#cv2").onclick=()=>{stopSweep();Object.assign(S.cv,{L:2,f:6,revealed:1e9,inMap:4});convToPeak();refresh();flyTo(viewFor());};
$("#cvIn").oninput=e=>{S.cv.inMap=+e.target.value;refresh();};

/* ================= Layer panels ================= */
const LYTXT={
 1:{badge:"Local · sees 3×3 px",title:"Layer 1: edges and strokes",intro:"Eight 3×3 filters slide over the raw pixels. A cell here sees a 3×3 speck of the image (about 1% of it), so the most it can report is “a bit of stroke at roughly this angle”. Every shape is made of strokes, so these channels fire for all five shapes.",
   pnote:"The 8 strongest responses in 1,500 test drawings. The teal box is the 3×3 the cell actually sees; the rest is context it never looks at."},
 2:{badge:"Mid-range · sees 8×8 px",title:"Layer 2: corners, crossings, curves",intro:"Twelve filters, each a 3×3 window stacked through all eight pooled layer-1 maps. Because pooling halved the grid, that window now spans 8×8 pixels: enough to see two strokes meet. Combining edge channels gives corners, crossings and bends.",
   pnote:"Strongest responses in the test drawings. Teal box: the 8×8 pixels the cell depends on."},
 3:{badge:"Near-global · sees 18×18 px",title:"Layer 3: whole parts of shapes",intro:"Sixteen filters on a 7×7 grid. Each cell now depends on an 18×18 patch, two thirds of the image, so it can hold a whole crossing or most of a square. Channels begin to belong to one shape, which is what the decision layer reads.",
   pnote:"Strongest responses in the test drawings; each patch is the full 18×18 receptive field."}};
function refreshLayer(L){
  const gi=LG[L],g=GR[gi],lay=M.layers[L-1],ch=S.ly[L],C=lay.channels[ch],t=LYTXT[L];
  $("#lyBadge").textContent=t.badge;$("#lyRf").textContent=`${g.n} channels · ${g.H}×${g.H}`;$("#lyTitle").textContent=t.title;$("#lyIntro").textContent=t.intro;
  const sl=$("#lyCh");sl.max=g.n-1;sl.value=ch;$("#lyChName").textContent=`${ch} · ${C.name}`;$("#lyDesc").textContent=C.desc;
  const A=groupData(gi).subarray(ch*g.H*g.H,(ch+1)*g.H*g.H);let mx=0,mi=0,sum=0;for(let i=0;i<A.length;i++){sum+=A[i];if(A[i]>mx){mx=A[i];mi=i;}}
  const means=[];for(let k=0;k<g.n;k++){let s=0;const B=groupData(gi);for(let i=0;i<g.H*g.H;i++)s+=B[k*g.H*g.H+i];means.push(s);}
  const rank=means.filter(v=>v>sum).length+1;
  drawMap($("#lyMini"),A,g.H,{under:true,mark:mx>0?[(mi/g.H)|0,mi%g.H,1]:null});
  $("#lyReadout").innerHTML=EMPTY?`<span class="k">draw something first</span>`:`<span class="k">peak</span> ${fmt(mx)} at row ${(mi/g.H)|0}, col ${mi%g.H}<br><span class="k">cells firing</span> ${A.filter(v=>v>.1*Math.max(mx,1e-6)&&v>0).length} of ${A.length}<br><span class="k">rank on this drawing</span> ${rank} of ${g.n}`;
  // weights
  const kw=$("#lyKernel");
  if(L===1){const w=W1.slice(ch*9,ch*9+9),m=Math.max(...w.map(Math.abs));$("#lyKTitle").textContent="Its 9 weights (sum to zero)";
    kw.innerHTML=`<div class="kgrids"><div class="kg"><div class="g">${w.map(v=>`<span style="${heatCell(v,m,CS.accent,CS.neg)}">${fmt(v,2)}</span>`).join("")}</div></div><p class="note" style="flex:1;min-width:140px">Teal weights reward ink, violet weights punish it. Where they sit decides which edge the filter likes. Bias ${fmt(B1[ch],2)}.</p></div>`;}
  else{const Cin=L===2?8:12,W=L===2?W2:W3,cvs=document.createElement("canvas");$("#lyKTitle").textContent=`Its ${Cin*9} weights: one 3×3 slice per input map`;
    const cell=12,gap=6,cols=Cin;cvs.width=cols*(3*cell+gap);cvs.height=3*cell+16;cvs.style.width="100%";cvs.style.maxWidth=cvs.width+"px";
    const x=cvs.getContext("2d");let m=1e-6;for(let i=0;i<Cin*9;i++)m=Math.max(m,Math.abs(W[ch*Cin*9+i]));
    for(let k=0;k<Cin;k++)for(let i=0;i<9;i++){const v=W[(ch*Cin+k)*9+i]/m;x.fillStyle=CS.bg;x.fillRect(k*(3*cell+gap)+(i%3)*cell,((i/3)|0)*cell,cell-1,cell-1);x.fillStyle=hexA(v>=0?CS.accent:CS.neg,Math.abs(v));x.fillRect(k*(3*cell+gap)+(i%3)*cell,((i/3)|0)*cell,cell-1,cell-1);}
    x.fillStyle=CS.muted;x.font="10px JetBrains Mono, monospace";for(let k=0;k<Cin;k++)x.fillText(String(k),k*(3*cell+gap)+12,3*cell+12);
    kw.innerHTML="";kw.appendChild(cvs);const p=document.createElement("p");p.className="note";p.textContent=`Slice k says how much this channel wants layer-${L-1} map k at each of the 9 spots. Teal = wants it, violet = wants it absent.`;kw.appendChild(p);}
  // tuning
  $("#lyTuneWrap").hidden=L!==1;if(L===1)drawTuning(ch);
  // patches
  const pw=$("#lyPatches");pw.innerHTML="";const s=lay.patchSize,mg=lay.margin,ink=hexToRgb(CS.ink),bg=hexToRgb(CS.bg);
  C.patches.forEach((b64,n)=>{const raw=atob(b64),cvs=document.createElement("canvas"),z=Math.max(4,Math.ceil(72/s));cvs.width=cvs.height=s*z;const x=cvs.getContext("2d");
    for(let i=0;i<s*s;i++){const v=raw.charCodeAt(i)/255;x.fillStyle=`rgb(${bg[0]+(ink[0]-bg[0])*v},${bg[1]+(ink[1]-bg[1])*v},${bg[2]+(ink[2]-bg[2])*v})`;x.fillRect((i%s)*z,((i/s)|0)*z,z,z);}
    if(mg){x.strokeStyle=CS.accent;x.lineWidth=2;x.strokeRect(mg*z+1,mg*z+1,(s-2*mg)*z-2,(s-2*mg)*z-2);}
    cvs.title=`From a ${CLS[C.patchY[n]].toLowerCase()} drawing`;pw.appendChild(cvs);});
  $("#lyPNote").textContent=t.pnote;$("#lyAtlas").textContent=`See all ${g.n} layer-${L} filters side by side in the feature atlas →`;
  const top=C.top200,w=top.indexOf(Math.max(...top));$("#lyClass").innerHTML=barsHTML(top,w);
}
function drawTuning(ch){const cv=$("#tuning"),x=cv.getContext("2d"),W=cv.width,H=cv.height,T=M.tuning[ch],n=T.length,bw=W/n;
  x.clearRect(0,0,W,H);const best=T.indexOf(Math.max(...T));
  for(let i=0;i<n;i++){const v=T[i],h=(H-34)*v;x.fillStyle=i===best?CS.accent:CS.grid;x.fillRect(i*bw+5,H-30-h,bw-10,h);
    const a=i*15*Math.PI/180,cx=i*bw+bw/2,cy=H-14,r=9;x.strokeStyle=CS.fg;x.lineWidth=2;x.lineCap="round";x.beginPath();x.moveTo(cx-r*Math.cos(a),cy+r*Math.sin(a));x.lineTo(cx+r*Math.cos(a),cy-r*Math.sin(a));x.stroke();}}
$("#lyCh").oninput=e=>{const L=+S.stage[1];S.lyTouched[L]=1;S.ly[L]=+e.target.value;if(S.sel&&S.sel.gi===LG[L])S.sel.ch=S.ly[L];refresh();};

/* ================= Local → global panel ================= */
const RFROWS=[[1,"Conv 1",3],[2,"Pool 1",4],[3,"Conv 2",8],[4,"Pool 2",10],[5,"Conv 3",18],[6,"Global max",28]];
function refreshRF(){
  $("#rfScale").innerHTML=RFROWS.map(([gi,n,s])=>`<div class="s${gi===S.rfG?" on":""}" data-g="${gi}" role="button" tabindex="0"><span>${n}</span><span class="t"><i style="width:${(s/28*100).toFixed(1)}%"></i></span><b>${gi===6?"all":s+"×"+s}</b></div>`).join("");
  $$("#rfScale .s").forEach(el=>{const go=()=>{S.rfG=+el.dataset.g;S.sel=null;refresh();flyTo(viewFor());};el.onclick=go;el.onkeydown=e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();go();}};});
  const sel=rfSel(),g=GR[S.rfG],s=g.rf,area=s>=28?100:Math.round(s*s/784*100);
  const say={1:"one bit of stroke edge",2:"the same, with a little slack in position",3:"two strokes meeting: a corner or a crossing",4:"a corner, wherever it sits in a small area",5:"most of a shape: a whole crossing, three sides of a box",6:"the entire drawing: it keeps each channel's strongest response, wherever it was"}[S.rfG];
  $("#rfReadout").innerHTML=`<span class="ink">${g.label.split(" · ")[0]}</span><br><span class="k">each cell sees</span> ${S.rfG===6?"all 28×28":s+"×"+s} px · ${area}% of the image<br><span class="k">big enough to hold</span> ${say}`;
  if(!S.sel||S.sel.gi!==S.rfG){if(S.rfG===6)S.sel={gi:6,ch:0,r:0,c:0};else if(sel.r!=null)S.sel={gi:S.rfG,ch:0,r:sel.r,c:sel.c};}
  const pr=M.probe;$("#probeBars").innerHTML=[["Avg brightness",pr[0]],["Layer 1",pr[2]],["Layer 2",pr[3]],["Layer 3",pr[4]]].map(([n,v],k)=>`<div class="b${k===3?" win":""}"><span class="g">${k?k:"·"}</span><span>${n}</span><span class="t"><i style="width:${v*100}%"></i></span><em>${(v*100).toFixed(1)}%</em></div>`).join("");
  $("#probeBars").querySelectorAll(".b").forEach(b=>b.style.gridTemplateColumns="22px 110px minmax(0,1fr) 52px");
  $("#purityBars").innerHTML=M.layers.map((l,k)=>`<div class="b${k===2?" win":""}"><span class="g">${k+1}</span><span>Layer ${k+1}</span><span class="t"><i style="width:${l.purity*100}%"></i></span><em>${Math.round(l.purity*100)}%</em></div>`).join("");
}

/* ================= Decision panel ================= */
let decTop=null;
function refreshDec(){
  const cls=S.decCls??NET.win,names=M.layers[2].channels;
  $("#decBars").innerHTML=barsHTML(EMPTY?[0,0,0,0,0]:NET.prob,EMPTY?-1:NET.win,{sel:cls});
  $$("#decBars .b").forEach(b=>{b.tabIndex=0;b.setAttribute("role","button");const go=()=>{S.decCls=+b.dataset.k;S.decFocus=null;refresh();};b.onclick=go;b.onkeydown=e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();go();}};});
  $("#evTitle").textContent=`The votes for “${CLS[cls]}”`;
  const ct=[...Array(16)].map((_,k)=>({k,v:WD[cls*16+k]*NET.g[k]})).sort((a,b)=>Math.abs(b.v)-Math.abs(a.v));
  decTop=ct.slice(0,5);
  if(S.decFocus==null||!decTop.some(o=>o.k===S.decFocus))S.decFocus=decTop[0].k;
  const fi=NET.gi[S.decFocus];S.sel={gi:5,ch:S.decFocus,r:(fi/7)|0,c:fi%7};
  const mx=Math.max(...ct.map(o=>Math.abs(o.v)),1e-6);
  $("#contrib").innerHTML=ct.slice(0,6).map(o=>{const w=Math.abs(o.v)/mx*50;return `<button data-k="${o.k}" aria-pressed="${o.k===S.decFocus}"><span>${o.k} · ${names[o.k].name}</span><span class="t"><i style="${o.v>=0?`left:50%;width:${w}%;background:${CS.accent}`:`right:50%;width:${w}%;background:${CS.neg}`}"></i></span><em style="color:${o.v>=0?CS.accent:CS.neg}">${o.v>=0?"+":""}${fmt(o.v)}</em></button>`;}).join("");
  $$("#contrib button").forEach(b=>b.onclick=()=>{S.decFocus=+b.dataset.k;refresh();});
  const tot=ct.reduce((a,o)=>a+o.v,0),k=S.decFocus,w=WD[cls*16+k];
  $("#decReadout").innerHTML=`<span class="k">channel ${k}</span> ${names[k].name}<br><span class="k">best match</span> ${fmt(NET.g[k])} at row ${S.sel.r}, col ${S.sel.c}<br><span class="k">× weight for ${CLS[cls]}</span> ${fmt(w)} <span class="k">=</span> <span style="color:${w*NET.g[k]>=0?CS.accent:CS.neg}">${w*NET.g[k]>=0?"+":""}${fmt(w*NET.g[k])}</span><br><span class="k">score(${CLS[cls]})</span> = Σ 16 votes ${fmt(tot)} + bias ${fmt(BD[cls])} = <span class="hit">${fmt(NET.lg[cls])}</span><br><span class="k">softmax over 5 scores →</span> ${EMPTY?"–":pct(NET.prob[cls])}`;
  $("#openL3").textContent=`Open channel ${k} in layer 3 →`;
}
$("#openL3").onclick=()=>{S.ly[3]=S.decFocus;S.lyTouched[3]=1;S.sel=null;setStage("l3");};

/* ================= About ================= */
function refreshAbout(){const nP=W1.length+B1.length+W2.length+B2.length+W3.length+B3.length+WD.length+BD.length;
  $("#facts").innerHTML=`<div><b>${nP.toLocaleString()}</b><span>trained weights</span></div><div><b>${(M.testAcc*100).toFixed(1)}%</b><span>correct on ${M.testN.toLocaleString()} unseen drawings</span></div><div><b>3</b><span>convolution layers</span></div><div><b>3→18</b><span>pixels a cell sees, layer 1 → 3</span></div>`;}

/* ================= Stage switching & refresh ================= */
function strongest(gi){const g=GR[gi],A=groupData(gi),HH=g.H*g.H;let best=0,bv=-1;for(let k=0;k<g.n;k++){let m=0,mu=0;for(let i=0;i<HH;i++){m=Math.max(m,A[k*HH+i]);mu+=A[k*HH+i]/HH;}m-=mu;
  // prefer channels that are both strong here and typical of the predicted shape
  const sp=M.layers[{1:0,3:1,5:2}[gi]].channels[k].top200[NET.win];const v=m*(.3+sp);if(v>bv){bv=v;best=k;}}return best;}
function inkNear(r0,c0){let best=[14,14],bd=1e9;for(let i=0;i<784;i++)if(NET.x[i]>.5){const r=(i/28)|0,c=i%28,d=(r-r0)**2+(c-c0)**2;if(d<bd){bd=d;best=[r+.5,c+.5];}}return best;}
function setStage(name){
  if(S.cv.sweeping&&name!=="conv")stopSweep();
  S.stage=name;S.about=false;
  $$(".tabs button").forEach(b=>b.setAttribute("aria-selected",b.dataset.stage===name));
  if(name==="dec"){S.decCls=null;S.decFocus=null;}
  if(name.startsWith("l")){const L=+name[1];if(!S.sel||S.sel.gi!==LG[L])S.sel=null;}
  if(name==="conv"||name==="dec")S.sel=null;
  if(name==="conv"&&!S.cv.sweeping){S.cv.revealed=1e9;convToPeak();}
  if(/^l[123]$/.test(name)){const L=+name[1];if(!S.lyTouched[L])S.ly[L]=strongest(LG[L]);}
  if(name==="rf"&&!S.rfCenter)S.rfCenter=inkNear(7,7);
  showSections();refresh();flyTo(viewFor());
  $("#panel").scrollTop=0;}
function showSections(){const st=S.about?"about":S.stage;
  $$(".sec[data-sec]").forEach(s=>{const k=s.dataset.sec;s.hidden=!(k===st||(k==="layer"&&/^l[123]$/.test(st)));});
  $("#viewBlock").hidden=st==="draw"||st==="about";}
$$(".tabs button").forEach(b=>b.onclick=()=>setStage(b.dataset.stage));

function pills(){const sp=$$("#signal span");
  if(wave){const t=(performance.now()-wave.start)/wave.dur*(NG+.5);sp.forEach((s,i)=>{s.classList.toggle("on",i===Math.floor(t));s.classList.toggle("done",i<Math.floor(t));});return;}
  const st=S.stage,on={conv:S.cv.L===1?[0,1]:[2,3],l1:[1],l2:[3],l3:[5],dec:[6,7],rf:[S.rfG]}[st]||[];
  sp.forEach((s,i)=>{s.classList.toggle("on",on.includes(i));s.classList.remove("done");});}

function run(anim){
  const n=S.stim?{x:S.stim,empty:false}:normalize();EMPTY=n.empty;NET=forward(n.x);
  if(anim)startWave();refresh();}
function refresh(){
  if(!NET)return;
  const st=S.about?"about":S.stage;
  if(st==="draw")refreshDraw();
  else if(st==="conv")refreshConv();
  else if(st==="l1"||st==="l2"||st==="l3")refreshLayer(+st[1]);
  else if(st==="rf")refreshRF();
  else if(st==="dec")refreshDec();
  else refreshAbout();
  if(st!=="draw"&&st!=="about")drawPixelView();
  updateCells();updateOverlays();pills();
}

$$("[data-preset]").forEach(b=>b.onclick=()=>preset(+b.dataset.preset));
$("#clearBtn").onclick=()=>{S.stim=null;strokes=[];redrawPad();S.sel=null;run(false);};
$("#replayBtn").onclick=()=>{startWave();};
$("#follow").onchange=e=>{S.follow=e.target.checked;};

/* ================= Theme ================= */
function applyTheme(){readColors();redrawPad();if(GL)applyColors3D();refresh();if(!$("#atlas").hidden)renderAtlas();if(!$("#shower").hidden)renderShower();}
matchMedia("(prefers-color-scheme: dark)").addEventListener("change",applyTheme);

/* ================= Loop ================= */
function loop(now){
  requestAnimationFrame(loop);if(!GL)return;
  if(fly){const k=clamp((now-fly.start)/fly.dur,0,1),e=k<.5?2*k*k:1-Math.pow(-2*k+2,2)/2;camera.position.lerpVectors(fly.p0,fly.P,e);controls.target.lerpVectors(fly.t0,fly.T,e);if(k>=1)fly=null;}
  if(wave){const t=(now-wave.start)/wave.dur;
    if(t>=1.05){wave=null;orb.visible=false;updateCells();pills();}
    else{const xs=GR.map(g=>g.xm),f=clamp(t*(NG+.5)-.5,0,NG-1),i=Math.floor(f),x=xs[i]+(xs[Math.min(NG-1,i+1)]-xs[i])*(f-i);
      orb.position.set(x,0,0);orb.visible=true;updateCells();pills();
      if(S.follow){controls.target.lerp(new THREE.Vector3(x,0,0),.08);camera.position.lerp(new THREE.Vector3(x-9,5,12),.05);}}}
  const lf=clamp(30/camera.position.distanceTo(controls.target),.62,1);
  if(Math.abs(lf-LF)>.005){LF=lf;labels.concat(dyn).forEach(sp=>sp.scale.set(sp.userData.bx*LF,sp.userData.size*LF,1));}
  controls.update();renderer.render(scene,camera);}

/* ================= Feature atlas (full-page view at #atlas) ================= */
const FV=JSON.parse($("#fv").textContent);
const dec64=b=>{const r=atob(b),a=new Uint8Array(r.length);for(let i=0;i<r.length;i++)a[i]=r.charCodeAt(i);return a;};
function paintGray(cv,bytes,n,opts={}){ // ink-on-paper rendering of an n×n byte image
  const z=opts.z||Math.max(3,Math.ceil(96/n));cv.width=cv.height=n*z;const x=cv.getContext("2d"),ink=hexToRgb(opts.col||CS.ink),bg=hexToRgb(CS.bg);
  for(let i=0;i<n*n;i++){const v=bytes[i]/255;x.fillStyle=`rgb(${bg[0]+(ink[0]-bg[0])*v|0},${bg[1]+(ink[1]-bg[1])*v|0},${bg[2]+(ink[2]-bg[2])*v|0})`;x.fillRect((i%n)*z,((i/n)|0)*z,z,z);}
  if(opts.box){const [o,s]=opts.box;x.strokeStyle=CS.accent;x.lineWidth=2;x.setLineDash([5,3]);x.strokeRect(o*z+1,o*z+1,s*z-2,s*z-2);x.setLineDash([]);}
  return cv;}
const mk=(tag,cls,html)=>{const e=document.createElement(tag);if(cls)e.className=cls;if(html!=null)e.innerHTML=html;return e;};

/* ---- test patterns ---- */
const STIM=[
  {cat:0,name:"flat line",g:"—",p:[[[60,140],[220,140]]]},
  {cat:0,name:"upright line",g:"|",p:[[[140,60],[140,220]]]},
  {cat:0,name:"rising line",g:"/",p:[[[83,197],[197,83]]]},
  {cat:0,name:"falling line",g:"\\",p:[[[83,83],[197,197]]]},
  {cat:1,name:"small ring",g:"∘",ring:[140,140,30]},
  {cat:1,name:"ring",g:"○",ring:[140,140,62]},
  {cat:1,name:"large ring",g:"◯",ring:[140,140,100]},
  {cat:1,name:"arch",g:"∩",arc:[140,180,75,Math.PI,2*Math.PI]},
  {cat:1,name:"bowl",g:"∪",arc:[140,100,75,0,Math.PI]},
  {cat:2,name:"corner ┌",g:"┌",p:[[[80,200],[80,80],[200,80]]]},
  {cat:2,name:"corner ┐",g:"┐",p:[[[80,80],[200,80],[200,200]]]},
  {cat:2,name:"corner └",g:"└",p:[[[80,80],[80,200],[200,200]]]},
  {cat:2,name:"corner ┘",g:"┘",p:[[[200,80],[200,200],[80,200]]]},
  {cat:3,name:"T-junction",g:"┬",p:[[[60,90],[220,90]],[[140,90],[140,215]]]},
  {cat:3,name:"plus crossing",g:"+",p:[[[60,140],[220,140]],[[140,60],[140,220]]]},
  {cat:3,name:"diagonal crossing",g:"×",p:[[[75,75],[205,205]],[[205,75],[75,205]]]},
  {cat:3,name:"peak",g:"∧",p:[[[70,200],[140,80],[210,200]]]},
  {cat:3,name:"valley",g:"∨",p:[[[70,80],[140,200],[210,80]]]},
  {cat:4,name:"dot",g:"•",dot:[140,140,20]}];
const CATS=["straight lines","curves and rings","corners","crossings and junctions","a dot"];
const CATSHORT=["Lines","Curves","Corners","Junctions","Dot"];
const sc=document.createElement("canvas");sc.width=sc.height=280;const sx=sc.getContext("2d",{willReadFrequently:true});
function stimImage(st){
  sx.fillStyle="#000";sx.fillRect(0,0,280,280);sx.strokeStyle="#fff";sx.fillStyle="#fff";sx.lineWidth=22;sx.lineCap="round";sx.lineJoin="round";
  if(st.p)st.p.forEach(pl=>{sx.beginPath();pl.forEach((q,i)=>i?sx.lineTo(q[0],q[1]):sx.moveTo(q[0],q[1]));sx.stroke();});
  if(st.ring){sx.beginPath();sx.arc(st.ring[0],st.ring[1],st.ring[2],0,2*Math.PI);sx.stroke();}
  if(st.arc){sx.beginPath();sx.arc(st.arc[0],st.arc[1],st.arc[2],st.arc[3],st.arc[4]);sx.stroke();}
  if(st.dot){sx.beginPath();sx.arc(st.dot[0],st.dot[1],st.dot[2],0,2*Math.PI);sx.fill();}
  const d=sx.getImageData(0,0,280,280).data,x=new Float32Array(784);
  for(let y=0;y<280;y++)for(let q=0;q<280;q++)x[((y/10)|0)*28+((q/10)|0)]+=d[(y*280+q)*4]/255/100;
  return x;}
let STIMX=null,PROBE=null,PROBE_METRIC="auto";window.__probe=(m)=>{PROBE_METRIC=m;computeProbe();return [1,2,3].map(L=>PROBE[L].map(r=>STIM[r.indexOf(Math.max(...r))].name).join(', '));};
function computeProbe(){ // strongest response of every channel to every pattern
  STIMX=STIM.map(stimImage);
  const raw={1:[],2:[],3:[]};
  STIMX.forEach((x,si)=>{const n=forward(x);let ink=0;for(let i=0;i<784;i++)ink+=x[i];
    [[1,n.a1,8,784],[2,n.a2,12,196],[3,n.a3,16,49]].forEach(([L,A,C,HH])=>{for(let k=0;k<C;k++){let m=0,s=0;for(let i=0;i<HH;i++){m=Math.max(m,A[k*HH+i]);s+=A[k*HH+i];}
      const met=PROBE_METRIC==="auto"?(L===1?"density":"max"):PROBE_METRIC;const v=met==="max"?m:met==="sum"?s:s/ink*HH/784;(raw[L][k]=raw[L][k]||[])[si]=v;}});});
  PROBE={};for(const L of [1,2,3])PROBE[L]=raw[L].map(row=>{const mx=Math.max(...row,1e-6);return row.map(v=>v/mx);});}
function loadStimulus(si,stage,L,k){
  S.stim=STIMX[si].slice();strokes=[];S.sel=null;
  if(L){S.ly[L]=k;S.lyTouched[L]=1;}
  location.hash="#explore";redrawPad();run(true);setStage(stage||"draw");}

/* ---- knock-out state ---- */
const KO={1:new Set(),2:new Set(),3:new Set()};

const GROUP={
 1:{title:"Edge and stroke detectors",rfTxt:"Local · each cell sees 3×3 px",
    txt:"Each filter is tuned to strokes at one angle. You can read the angle straight off its typical input: \\, |, / or —. Every shape is made of strokes, so these filters fire for all five shapes and on their own say little about which shape it is."},
 2:{title:"Corner, curve and junction detectors",rfTxt:"Mid-range · each cell sees 8×8 px",
    txt:"Each filter combines a few layer-1 edges that sit next to each other. A flat stroke meeting an upright one makes a corner ┐; a / meeting a \\ over the top makes an arch ∩. Curves first appear here, because an 8×8 view is big enough to see a stroke bend."},
 3:{title:"Shape-part detectors",rfTxt:"Near-global · each cell sees 18×18 px",
    txt:"With an 18×18 view a filter can hold a whole X, most of a triangle or the top of a box. These are the filters the final vote listens to, and the only layer where losing a single filter can break the network."}};

function renderAtlas(){
  if(!NET)return;
  computeProbe();
  const body=$("#atlasBody");body.innerHTML="";
  // how to read
  const how=mk("section","gs wide",`<div class="txt"><span class="num">How to read a filter card</span>
    <div class="howrow"><div><b>Typical input</b><span>The average of the 50 test patches that excite this filter most. The dashed box is exactly what one cell sees.</span></div>
    <div><b>Synthesised</b><span>An input built from scratch by gradient ascent to excite it as much as possible: the pattern it is “looking for”.</span></div>
    <div><b>Shapes</b><span>Which of the five shapes those strongest responses came from.</span></div>
    <div><b>Without it</b><span>Test accuracy when this one filter is switched off (1,500 drawings).</span></div></div></div>`);
  body.appendChild(how);
  for(const L of [1,2,3])body.appendChild(layerSection(L));
  body.appendChild(probeSection());
  body.appendChild(knockSection());
  body.appendChild(dreamSection());
  renderKO();
}

function layerSection(L){
  const lay=M.layers[L-1],C=lay.channels.length,G=GROUP[L],ab=FV.ablate[L],base=FV.base;
  let worst=0;ab.forEach(r=>r.forEach((v,c)=>worst=Math.max(worst,base[c]-v)));
  const sec=mk("section","gs wide");sec.id="atlas-l"+L;
  sec.appendChild(mk("div","txt",`<span class="num">Layer ${L} · ${G.rfTxt}</span><h2>${G.title}</h2><p>${G.txt}</p>`));
  sec.appendChild(mk("div","stats",`<div><b>${C}</b><span>filters</span></div><div><b>${[3,8,18][L-1]}×${[3,8,18][L-1]}</b><span>pixels one cell sees</span></div>
    <div><b>${(M.probe[L+1]*100).toFixed(0)}%</b><span>shape guessed from this layer alone</span></div><div><b>${(worst*100).toFixed(worst<.1?1:0)}%</b><span>most accuracy lost by removing any one filter</span></div>`));
  const grid=mk("div","tiles");
  lay.channels.forEach((ch,k)=>{
    const t=mk("article","tile");
    const top=mk("div","tvis");
    const mean=FV.mean[L],opt=FV.opt[L];
    const c1=paintGray(document.createElement("canvas"),dec64(mean.img[k]),mean.size,{box:[mean.margin,mean.size-2*mean.margin]});c1.className="big";c1.setAttribute("aria-label",`Typical input for filter ${k}`);
    const c2=paintGray(document.createElement("canvas"),dec64(opt.img[k]),opt.size,{col:CS.accent});c2.className="small";c2.setAttribute("aria-label",`Synthesised input for filter ${k}`);
    const w1=mk("figure","",null);w1.appendChild(c1);w1.appendChild(mk("figcaption","","typical"));
    const w2=mk("figure","",null);w2.appendChild(c2);w2.appendChild(mk("figcaption","","synthesised"));
    top.appendChild(w1);top.appendChild(w2);t.appendChild(top);
    t.appendChild(mk("h4","",`<span>${L}·${k}</span> ${ch.name}`));
    t.appendChild(mk("p","tdesc",ch.desc));
    const sh=ch.top200,mx=Math.max(...sh);
    t.appendChild(mk("div","shares",sh.map((v,c)=>`<span class="${v===mx?"on":""}" title="${CLS[c]} ${Math.round(v*100)}%"><i style="height:${Math.max(2,v/mx*26)}px"></i>${GLY[c]}</span>`).join("")));
    if(L===1){const cv=document.createElement("canvas");cv.width=240;cv.height=60;cv.className="tune";drawTuneSmall(cv,k);t.appendChild(cv);}
    let wc=0,wd=0;ab[k].forEach((v,c)=>{if(base[c]-v>wd){wd=base[c]-v;wc=c;}});
    t.appendChild(mk("p","ko"+(wd>.2?" bad":""),wd<.02?`<b>Without it:</b> barely matters, other filters cover for it`:`<b>Without it:</b> ${GLY[wc]} ${CLS[wc]} ${Math.round(base[wc]*100)}% → ${Math.round(ab[k][wc]*100)}% correct`));
    const b=mk("button","btn","Open in 3D →");b.onclick=()=>{S.ly[L]=k;S.lyTouched[L]=1;S.sel=null;location.hash="#explore";setStage("l"+L);};
    t.appendChild(b);grid.appendChild(t);});
  sec.appendChild(grid);return sec;}
function drawTuneSmall(cv,ch){const x=cv.getContext("2d"),W=cv.width,H=cv.height,T=M.tuning[ch],n=T.length,bw=W/n,best=T.indexOf(Math.max(...T));
  for(let i=0;i<n;i++){const h=(H-22)*T[i];x.fillStyle=i===best?CS.accent:CS.grid;x.fillRect(i*bw+3,H-20-h,bw-6,h);
    const a=i*15*Math.PI/180,cx=i*bw+bw/2,cy=H-9,r=6;x.strokeStyle=CS.muted;x.lineWidth=1.5;x.beginPath();x.moveTo(cx-r*Math.cos(a),cy+r*Math.sin(a));x.lineTo(cx+r*Math.cos(a),cy-r*Math.sin(a));x.stroke();}}

function probeSection(){
  const sec=mk("section","gs wide");sec.id="atlas-probe";
  // findings
  const finds=[1,2,3].map(L=>{const cnt=[0,0,0,0,0];PROBE[L].forEach(row=>cnt[STIM[row.indexOf(Math.max(...row))].cat]++);
    const parts=cnt.map((n,c)=>n?`${n} to ${CATS[c]}`:null).filter(Boolean);
    return `<li><b>Layer ${L}:</b> of ${PROBE[L].length} filters, ${parts.join(", ")}.</li>`;}).join("");
  const ringFans=[];[1,2,3].forEach(L=>PROBE[L].forEach((row,k)=>{const b=row.indexOf(Math.max(...row));if(STIM[b].cat===1)ringFans.push(`${L}·${k} (${STIM[b].name})`);}));
  const lineFans=[];[1,2,3].forEach(L=>PROBE[L].forEach((row,k)=>{const b=row.indexOf(Math.max(...row));if(STIM[b].cat===0)lineFans.push(`${L}·${k} (${STIM[b].name})`);}));
  sec.appendChild(mk("div","txt",`<span class="num">Probe lab · controlled experiments</span><h2>Which filters like straight lines, which like curves?</h2>
    <p>Each column is a simple test pattern fed straight into the network. Each cell is how strongly one filter responds to it, scaled so every row's favourite is full teal and outlined. Read across a row to see what one filter prefers; read down a column to see which filters answer a pattern. Click a cell to load that pattern into the 3D explorer and open that filter.</p>
    <p class="note">How “response” is measured: for layer 2 and 3, a filter's strongest cell anywhere, which is what the network passes on. For layer 1, its average response along the ink: a ring contains a short piece at every angle, so a 3×3 filter's single strongest cell can't tell a ring from a line, but its average shows which angle it is tuned to.</p>
    <div class="rule"><b>Each filter's favourite pattern</b><ul class="finds">${finds}</ul>Favour a curve or ring: ${ringFans.join(", ")||"none"}.<br>Favour a straight line: ${lineFans.join(", ")||"none"}.</div>`));
  const wrap=mk("div","mwrap"),m=mk("div","matrix");m.style.gridTemplateColumns=`210px repeat(${STIM.length},34px)`;
  // category header
  m.appendChild(mk("div","mh",""));
  let prev=-1;STIM.forEach((st,si)=>{const c=mk("div","mcat"+(st.cat!==prev?" first":""),st.cat!==prev?CATSHORT[st.cat]:"");prev=st.cat;m.appendChild(c);});
  m.appendChild(mk("div","mh","Filter ↓ · pattern →"));
  STIM.forEach((st,si)=>{const b=mk("button","mstim");b.title=`${st.name}: load it into the explorer`;
    const cv=paintGray(document.createElement("canvas"),Uint8Array.from(STIMX[si],v=>Math.min(255,v*255)),28,{z:2,col:CS.fg});b.appendChild(cv);
    b.onclick=()=>loadStimulus(si,"draw");m.appendChild(b);});
  [1,2,3].forEach(L=>{
    const hdr=mk("div","mlayer",`Layer ${L} · sees ${[3,8,18][L-1]}×${[3,8,18][L-1]} px`);hdr.style.gridColumn=`1 / span ${STIM.length+1}`;m.appendChild(hdr);
    PROBE[L].forEach((row,k)=>{
      m.appendChild(mk("div","mlab",`<span>${L}·${k}</span> ${M.layers[L-1].channels[k].name}`));
      const best=row.indexOf(Math.max(...row));
      row.forEach((v,si)=>{const c=mk("button","mcell"+(si===best?" best":""));c.style.background=hexA(CS.accent,.06+.94*v*v);c.title=`${L}·${k} ${M.layers[L-1].channels[k].name} × ${STIM[si].name}: ${Math.round(v*100)}%`;
        c.setAttribute("aria-label",c.title);c.onclick=()=>loadStimulus(si,"l"+L,L,k);m.appendChild(c);});});});
  wrap.appendChild(m);sec.appendChild(wrap);
  sec.appendChild(mk("p","note","Patterns are drawn at 28×28 with the same stroke width as the training drawings, centred, and are not rescaled. Layer-3 filters were trained on whole shapes, so their responses to these single parts are weaker and less tidy."));
  return sec;}

function knockSection(){
  const sec=mk("section","gs wide");sec.id="atlas-ko";
  const base=FV.base,ab=FV.ablate;
  const worst=[0,1,2,3,4].map(c=>{let b={L:1,k:0,v:1};[1,2,3].forEach(L=>ab[L].forEach((r,k)=>{if(r[c]<b.v)b={L,k,v:r[c]};}));return b;});
  const tbl=worst.map((b,c)=>`<div class="krow"><span class="g">${GLY[c]}</span><span>${CLS[c]}</span><span>without <b>${b.L}·${b.k} ${M.layers[b.L-1].channels[b.k].name}</b></span><em>${Math.round(base[c]*100)}% → <b class="${b.v<.6?"neg":""}">${Math.round(b.v*100)}%</b></em></div>`).join("");
  sec.appendChild(mk("div","txt",`<span class="num">Knock-out experiment</span><h2>Switch filters off and see what breaks</h2>
    <p>Removing a filter is the most direct test of what it is for. Click filters below to switch them off, and watch the prediction for the current drawing. Early filters are redundant: lose one and its neighbours cover. Layer 3 has single points of failure.</p>`));
  const live=mk("div","kogrid");
  live.innerHTML=`<div class="kleft"><div class="row"><canvas id="koIn" width="28" height="28" aria-label="Current input"></canvas><div class="kex">${GLY.map((g,c)=>`<button class="btn glyph" data-c="${c}" aria-label="Use a ${CLS[c].toLowerCase()}">${g}</button>`).join("")}</div></div>
    <div id="koChips"></div><div class="row"><button class="btn" id="koReset">Switch all back on</button><span class="cap" id="koCount"></span></div></div>
    <div class="kright"><h3>Prediction for this drawing</h3><div class="kbars" id="koBars"></div><p class="note">Outline: with every filter on. Solid: with your filters switched off.</p></div>`;
  sec.appendChild(live);
  sec.appendChild(mk("div","",`<h3 style="margin-top:6px">Measured on 1,500 test drawings: the single filter each shape can least afford to lose</h3><div class="ktable">${tbl}</div>`));
  setTimeout(()=>{
    $$("#atlas-ko .kex button").forEach(b=>b.onclick=()=>{preset(+b.dataset.c);renderKO();});
    $("#koReset").onclick=()=>{[1,2,3].forEach(L=>KO[L].clear());renderKO();};},0);
  return sec;}
function renderKO(){
  if(!$("#koChips"))return;
  $("#koChips").innerHTML=[1,2,3].map(L=>`<div class="kl"><span class="cap">Layer ${L}</span><div class="kchips">${M.layers[L-1].channels.map((ch,k)=>`<button data-l="${L}" data-k="${k}" aria-pressed="${KO[L].has(k)}" title="${ch.name}">${k}</button>`).join("")}</div></div>`).join("");
  $$("#koChips button").forEach(b=>b.onclick=()=>{const L=+b.dataset.l,k=+b.dataset.k;KO[L].has(k)?KO[L].delete(k):KO[L].add(k);renderKO();});
  const n=KO[1].size+KO[2].size+KO[3].size;$("#koCount").textContent=n?`${n} filter${n>1?"s":""} off`:"all 36 filters on";
  const on=NET,off=forward(NET.x,KO);
  $("#koBars").innerHTML=CLS.map((c,i)=>{const w=off.win===i;return `<div class="b${w?" win":""}"><span class="g">${GLY[i]}</span><span>${c}</span><span class="t"><i style="width:${(off.prob[i]*100).toFixed(1)}%"></i><u style="width:${(on.prob[i]*100).toFixed(1)}%"></u></span><em>${pct(off.prob[i])}</em></div>`;}).join("");
  const cv=$("#koIn"),x=cv.getContext("2d"),im=x.createImageData(28,28),bg=hexToRgb(CS.bg),fg=hexToRgb(CS.ink);
  for(let i=0;i<784;i++){const v=NET.x[i];im.data.set([bg[0]+(fg[0]-bg[0])*v,bg[1]+(fg[1]-bg[1])*v,bg[2]+(fg[2]-bg[2])*v,255],i*4);}x.putImageData(im,0,0);}

function dreamSection(){
  const sec=mk("section","gs wide");
  sec.appendChild(mk("div","txt",`<span class="num">What the network “imagines”</span><h2>Inputs that maximise each shape's score</h2>
    <p>Gradient ascent on a blank 28×28 image, pushing one class score as high as it will go. The results show what the network checks for, not what a shape looks like: a crossing for Cross, slanted strokes meeting a base for Triangle, scattered curve pieces for Circle. It only needs the right parts to be present somewhere, because the global max throws position away.</p>`));
  const row=mk("div","dreams");
  FV.cls.forEach((b,c)=>{const f=mk("figure","");const cv=paintGray(document.createElement("canvas"),dec64(b),28,{z:5,col:CS.accent});f.appendChild(cv);f.appendChild(mk("figcaption","",`${GLY[c]} ${CLS[c]}`));row.appendChild(f);});
  sec.appendChild(row);return sec;}


/* ================= Shower lab (#shower): back-propagation by analogy ================= */
const TCOLD=20,THOT=60,START={a:.5,c:.25,b:.6};
const SH={mode:2,share:false,a:START.a,c:START.c,b:START.b,target:38,lr:.001,hist:[],stage:-1,timer:null,auto:null,steps:0,built:null};
/* the deep shower: 3 mixers → 2 valves → 1 output valve (3 layers, 6 knobs). Mixer 2 feeds both valves. */
const DSTART={a1:.8,a2:.4,a3:.2,b1:.5,b2:.5,c:.4},K3=["a1","a2","a3","b1","b2","c"];SH.D={...DSTART};
function dForward(){const D=SH.D,T1=[D.a1,D.a2,D.a3].map(a=>TCOLD+40*a),T2=[D.b1*T1[0]+(1-D.b1)*T1[1],D.b2*T1[1]+(1-D.b2)*T1[2]];return {T1,T2,T:D.c*T2[0]+(1-D.c)*T2[1]};}
function dGrad(f){const D=SH.D,g2=[D.c,1-D.c],p=[[g2[0]*D.b1],[g2[0]*(1-D.b1),g2[1]*D.b2],[g2[1]*(1-D.b2)]],g1=p.map(x=>x.reduce((s,v)=>s+v,0));
  return {c:f.T2[0]-f.T2[1],b1:g2[0]*(f.T1[0]-f.T1[1]),b2:g2[1]*(f.T1[1]-f.T1[2]),a1:40*g1[0],a2:40*g1[1],a3:40*g1[2],g2,g1,p};}
const getK=k=>SH.mode===3?SH.D[k]:SH[k];
function shForward(){
  if(SH.mode===3)return dForward();
  if(SH.mode===1){const T=TCOLD+(THOT-TCOLD)*SH.a;return {TA:T,T};}
  const c=SH.share?SH.a:SH.c,TA=TCOLD+40*SH.a,TB=TCOLD+40*c;return {TA,TB,T:SH.b*TA+(1-SH.b)*TB};}
function shGrad(f){ // ∂T/∂knob for every knob, by the chain rule
  if(SH.mode===3)return dGrad(f);
  if(SH.mode===1)return {a:40};
  const gA=SH.b*40,gC=(1-SH.b)*40,gb=f.TA-f.TB;
  return SH.share?{a:gA+gC,b:gb,gA,gC}:{a:gA,c:gC,b:gb,gA,gC};}
function shDeltas(){const f=shForward(),e=f.T-SH.target,g=shGrad(f),d={};
  for(const k of (SH.mode===3?K3:["a","b","c"]))if(g[k]!=null)d[k]=-SH.lr*e*g[k];return {f,e,g,d};}
function shStep(){const {d}=shDeltas(),o=SH.mode===3?SH.D:SH;for(const k in d)o[k]=clamp(o[k]+d[k],0,1);SH.steps++;SH.hist.push(shForward().T);SH.stage=-1;}
const tempCol=T=>{const t=clamp((T-TCOLD)/(THOT-TCOLD),0,1),c=hexToRgb(CS.neg),h=hexToRgb(CS.ink);return `rgb(${c[0]+(h[0]-c[0])*t|0},${c[1]+(h[1]-c[1])*t|0},${c[2]+(h[2]-c[2])*t|0})`;};
const sgn=(v,d=3)=>(v>=0?"+":"−")+Math.abs(v).toFixed(d);

/* ---- the plumbing picture ---- */
const PIPES={hotA:"M130,60 C205,60 200,122 236,127",coldA:"M130,380 C190,380 175,150 237,138",hotB:"M130,60 C190,60 175,292 237,302",coldB:"M130,380 C205,380 200,318 236,313",
  outA:"M304,130 C405,130 410,206 463,213",outB:"M304,310 C405,310 410,234 463,227",out:"M538,220 H628",direct:"M304,130 C440,130 500,220 628,220"};
function knobSVG(id,x,y,label,sc=1){return `<g class="knob" id="k_${id}" data-k="${id}" tabindex="0" role="slider" aria-label="${label}" transform="translate(${x},${y}) scale(${sc})">
  <circle r="36" class="kface"/><path class="ktrack" d="${arcPath(28,-135,135)}"/><path class="kfill" d=""/><line class="kptr" x1="0" y1="0" x2="0" y2="-24"/><circle r="5" class="khub"/>
  <text class="klab" y="54">${label}</text><text class="kval" y="70"></text></g>`;}
function arcPath(r,a0,a1){const p=a=>[r*Math.sin(a*Math.PI/180),-r*Math.cos(a*Math.PI/180)];const [x0,y0]=p(a0),[x1,y1]=p(a1);return `M${x0.toFixed(1)},${y0.toFixed(1)} A${r},${r} 0 ${a1-a0>180?1:0} 1 ${x1.toFixed(1)},${y1.toFixed(1)}`;}
const pill=(id,x,y)=>`<g class="pill" id="p_${id}" transform="translate(${x},${y})" opacity="0"><rect rx="11" height="22" y="-11"/><text y="4"></text></g>`;
const TANKS=(hy=30,cy=350)=>`<g class="tank"><rect x="20" y="${hy}" width="110" height="60" rx="10" class="thot"/><text x="75" y="${hy+28}">Hot</text><text x="75" y="${hy+46}" class="sub">${THOT} °C</text></g>
  <g class="tank"><rect x="20" y="${cy}" width="110" height="60" rx="10" class="tcold"/><text x="75" y="${cy+28}">Cold</text><text x="75" y="${cy+46}" class="sub">${TCOLD} °C</text></g>`;
const HEAD=(hx=628,hy=220)=>`<g id="head" transform="translate(${hx},${hy})"><path class="hpipe" d="M0,-6 H28 V-30"/><path class="hhead" d="M8,-30 H64 L56,-14 H16 Z"/>
    <g id="drops">${[0,1,2,3,4,5,6].map(i=>`<line class="drop" x1="${18+i*6}" y1="-6" x2="${10+i*8}" y2="${70+((i*13)%17)}"/>`).join("")}</g>
    <text id="shT" class="bigT" x="36" y="-46"></text><text id="shTarget" class="sub" x="36" y="112"></text><text id="shTarget2" class="sub" x="36" y="128"></text></g>`;
const DPIPES={hot1:"M130,70 C170,70 165,106 189,110",hot2:"M130,70 C180,70 160,254 189,257",hot3:"M130,70 C185,70 150,404 190,406",
  cold1:"M130,470 C185,470 150,120 190,114",cold2:"M130,470 C180,470 160,266 189,263",cold3:"M130,470 C170,470 165,414 189,412",
  m1v1:"M244,110 C310,110 325,178 372,182",m2v1:"M244,260 C310,260 325,192 372,188",m2v2:"M244,260 C310,260 325,328 372,332",m3v2:"M244,410 C310,410 325,342 372,338",
  v1o:"M428,185 C490,185 500,252 538,256",v2o:"M428,335 C490,335 500,268 538,264",out:"M594,260 H636"};
const DKN={a1:[215,110],a2:[215,260],a3:[215,410],b1:[400,185],b2:[400,335],c:[565,260]};
const DNUDGE={a1:[215,180],a2:[215,330],a3:[215,480],b1:[400,148],b2:[400,298],c:[565,366]};
function buildDeep(){
  const svg=$("#shSvg");svg.setAttribute("viewBox","0 0 760 520");
  svg.innerHTML=`${TANKS(40,440)}
  <text class="lhead" x="215" y="22">LAYER 1 · mixers</text><text class="lhead" x="400" y="22">LAYER 2 · valves</text><text class="lhead" x="565" y="22">LAYER 3 · output</text>
  ${Object.entries(DPIPES).map(([k,d])=>`<path id="pp_${k}" class="pipe${k.startsWith("hot")||k.startsWith("cold")?" thin":""}" d="${d}"/><path id="pw_${k}" class="water" d="${d}"/><path id="pf_${k}" class="flow" d="${d}" opacity="0"/>`).join("")}
  ${HEAD(636,260)}
  ${K3.map(k=>knobSVG(k,DKN[k][0],DKN[k][1],{a1:"mixer a₁",a2:"mixer a₂",a3:"mixer a₃",b1:"valve b₁",b2:"valve b₂",c:"valve c"}[k],.78)).join("")}
  ${pill("err",612,228)}${pill("c",565,340)}${pill("vc1",488,214)}${pill("vc2",488,306)}${pill("b1",400,118)}${pill("b2",400,414)}
  ${pill("m11",318,145)}${pill("m21",318,225)}${pill("m22",318,295)}${pill("m32",318,375)}${pill("a1",215,58)}${pill("a2",215,208)}${pill("a3",215,358)}
  ${K3.map(k=>`<text id="n_${k}" class="nudge mid" x="${DNUDGE[k][0]}" y="${DNUDGE[k][1]}"></text>`).join("")}`;
  bindKnobs();SH.built="deep";}
function bindKnobs(){
  $$("#shSvg .knob").forEach(g=>{const k=g.dataset.k;let y0=null,v0=0;
    g.addEventListener("pointerdown",e=>{y0=e.clientY;v0=getK(k);g.setPointerCapture(e.pointerId);stopAuto();});
    g.addEventListener("pointermove",e=>{if(y0==null)return;setKnob(k,v0+(y0-e.clientY)/160);});
    g.addEventListener("pointerup",()=>{y0=null;});
    g.addEventListener("keydown",e=>{const d={ArrowUp:.02,ArrowRight:.02,ArrowDown:-.02,ArrowLeft:-.02}[e.key];if(d){e.preventDefault();setKnob(k,getK(k)+d);}});});}
function buildShower(){
  const svg=$("#shSvg");svg.setAttribute("viewBox","0 0 760 440");
  svg.innerHTML=`
  ${TANKS()}
  ${Object.entries(PIPES).map(([k,d])=>`<path id="pp_${k}" class="pipe" d="${d}"/><path id="pw_${k}" class="water" d="${d}"/><path id="pf_${k}" class="flow" d="${d}" opacity="0"/>`).join("")}
  ${HEAD()}
  ${knobSVG("a",270,130,"Mixer A · knob a")}${knobSVG("c",270,310,"Mixer B · knob c")}${knobSVG("b",500,220,"Valve · knob b")}
  <line id="shareLink" class="share" x1="270" y1="208" x2="270" y2="274"/>
  ${pill("err",586,248)}${pill("b",478,318)}${pill("pa",368,108)}${pill("pb",356,332)}${pill("a",270,80)}${pill("c",270,250)}
  <text id="n_a" class="nudge mid" x="270" y="224"></text><text id="n_c" class="nudge mid" x="270" y="402"></text><text id="n_b" class="nudge mid" x="500" y="172"></text>`;
  bindKnobs();SH.built="basic";}
function setKnob(k,v){if(SH.mode===3)SH.D[k]=clamp(v,0,1);else SH[k]=clamp(v,0,1);if(SH.share&&k==="c")SH.a=SH.c;SH.stage=-1;SH.hist=[shForward().T];SH.steps=0;drawShower();}

function setPill(id,txt,show,cls){const g=$("#p_"+id);if(!g)return;g.setAttribute("opacity",show?1:0);const t=g.querySelector("text"),r=g.querySelector("rect");t.textContent=txt;
  const w=Math.max(40,txt.length*7.3+18);r.setAttribute("width",w);r.setAttribute("x",-w/2);t.setAttribute("text-anchor","middle");g.setAttribute("class","pill"+(cls?" "+cls:""));}
function drawShower(){
  if(!SH.built)return;
  if(SH.mode===3){drawDeep();return;}
  const two=SH.mode===2,f=shForward(),{e,g,d}=shDeltas(),st=SH.stage;
  // visibility per mode
  const vis={hotB:two,coldB:two,outA:two,outB:two,out:two,direct:!two};
  for(const k in PIPES){const v=vis[k]!==false;["pp_","pw_","pf_"].forEach(p=>$("#"+p+k).style.display=v?"":"none");}
  $("#k_c").style.display=two?"":"none";$("#k_b").style.display=two?"":"none";$("#shareLink").style.display=two&&SH.share?"":"none";
  // water colours
  const set=(k,T)=>$("#pp_"+k).style.stroke=tempCol(T);
  set("hotA",THOT);set("hotB",THOT);set("coldA",TCOLD);set("coldB",TCOLD);set("outA",f.TA);set("direct",f.TA);
  if(two){set("outB",f.TB);set("out",f.T);}
  // flow width by how much water each pipe carries (valve split)
  if(two){$("#pp_outA").style.strokeWidth=4+10*SH.b;$("#pp_outB").style.strokeWidth=4+10*(1-SH.b);}
  $$("#drops .drop").forEach(l=>l.style.stroke=tempCol(f.T));$("#head .hhead").style.fill=tempCol(f.T);
  $("#shT").textContent=f.T.toFixed(1)+" °C";$("#shT").style.fill=tempCol(f.T);
  $("#shTarget").textContent=`want ${SH.target.toFixed(0)} °C`;$("#shTarget2").textContent=Math.abs(e)<.05?"just right":e<0?(-e).toFixed(1)+"° too cold":e.toFixed(1)+"° too hot";
  // knobs
  [["a",SH.a,`a = ${SH.a.toFixed(3)}`],["c",SH.share?SH.a:SH.c,SH.share?"shares knob a":`c = ${SH.c.toFixed(3)}`],["b",SH.b,`b = ${SH.b.toFixed(3)} to A`]].forEach(([k,v,txt])=>{
    const el=$("#k_"+k);el.querySelector(".kptr").setAttribute("transform",`rotate(${-135+270*v})`);el.querySelector(".kfill").setAttribute("d",v>.004?arcPath(28,-135,-135+270*v):"");
    el.querySelector(".kval").textContent=txt;el.setAttribute("aria-valuenow",v.toFixed(2));el.classList.toggle("hot",st>=2&&(k!=="b"||st>=1));});
  $("#k_a .klab").textContent=two?(SH.share?"Shared knob a":"Mixer A · knob a"):"The only knob · a";
  // backward-pass reveal, stage by stage
  const two_=two;
  setPill("err",`error ${e>=0?"+":"−"}${Math.abs(e).toFixed(2)}°`,st>=0,"e");
  setPill("b",`∂T/∂b = TA − TB = ${g.b!=null?g.b.toFixed(1):""}`,two_&&st>=1);
  setPill("pa",`× b = ${SH.b.toFixed(2)}`,two_&&st>=1,"m");
  setPill("pb",`× (1 − b) = ${(1-SH.b).toFixed(2)}`,two_&&st>=1,"m");
  setPill("a",two_?(SH.share?`∂T/∂a = ${g.gA.toFixed(1)} + ${g.gC.toFixed(1)} = ${g.a.toFixed(1)}`:`∂T/∂a = ${SH.b.toFixed(2)} × 40 = ${g.a.toFixed(1)}`):`∂T/∂a = 40`,st>=(two_?2:1));
  setPill("c",`∂T/∂c = ${(1-SH.b).toFixed(2)} × 40 = ${g.c!=null?g.c.toFixed(1):""}`,two_&&!SH.share&&st>=2);
  const showN=st>=(two_?3:2);
  [["a",d.a],["c",SH.share?null:d.c],["b",d.b]].forEach(([k,v])=>{const t=$("#n_"+k);t.textContent=showN&&v!=null&&(two_||k==="a")?`${v>=0?"↻ turn up":"↺ turn down"} ${sgn(v)}`:"";});
  // reverse-flow overlays
  const flows=two_?{out:st>=0,outA:st>=1,outB:st>=1}:{direct:st>=0};
  for(const k in PIPES)$("#pf_"+k).setAttribute("opacity",flows[k]?1:0);
  shLedger(f,e,g,d);shChart();
  $("#shBack").textContent=SH.stage<0?"◀ Backward pass":SH.stage>=(two_?3:2)?"◀ Backward pass (done)":"◀ Next backward step";
}
function setKnobVis(k,v,txt,hot){const el=$("#k_"+k);if(!el)return;el.querySelector(".kptr").setAttribute("transform",`rotate(${-135+270*v})`);
  el.querySelector(".kfill").setAttribute("d",v>.004?arcPath(28,-135,-135+270*v):"");el.querySelector(".kval").textContent=txt;el.setAttribute("aria-valuenow",v.toFixed(2));el.classList.toggle("hot",!!hot);}
function drawDeep(){
  const D=SH.D,f=dForward(),{e,g,d}=shDeltas(),st=SH.stage,f2=v=>v.toFixed(2);
  const set=(k,T,w)=>{const p=$("#pp_"+k);p.style.stroke=tempCol(T);if(w!=null)p.style.strokeWidth=w;};
  ["hot1","hot2","hot3"].forEach(k=>set(k,THOT));["cold1","cold2","cold3"].forEach(k=>set(k,TCOLD));
  // pipe width ∝ share of the final flow each pipe carries
  const W=x=>3+11*x;
  set("m1v1",f.T1[0],W(D.c*D.b1));set("m2v1",f.T1[1],W(D.c*(1-D.b1)));set("m2v2",f.T1[1],W((1-D.c)*D.b2));set("m3v2",f.T1[2],W((1-D.c)*(1-D.b2)));
  set("v1o",f.T2[0],W(D.c));set("v2o",f.T2[1],W(1-D.c));set("out",f.T,W(1));
  $$("#drops .drop").forEach(l=>l.style.stroke=tempCol(f.T));$("#head .hhead").style.fill=tempCol(f.T);
  $("#shT").textContent=f.T.toFixed(1)+" °C";$("#shT").style.fill=tempCol(f.T);
  $("#shTarget").textContent=`want ${SH.target.toFixed(0)} °C`;$("#shTarget2").textContent=Math.abs(e)<.05?"just right":e<0?(-e).toFixed(1)+"° too cold":e.toFixed(1)+"° too hot";
  setKnobVis("a1",D.a1,`${f.T1[0].toFixed(1)} °C`,st>=3);setKnobVis("a2",D.a2,`${f.T1[1].toFixed(1)} °C`,st>=3);setKnobVis("a3",D.a3,`${f.T1[2].toFixed(1)} °C`,st>=3);
  setKnobVis("b1",D.b1,`${f.T2[0].toFixed(1)} °C`,st>=2);setKnobVis("b2",D.b2,`${f.T2[1].toFixed(1)} °C`,st>=2);setKnobVis("c",D.c,`${f.T.toFixed(1)} °C`,st>=1);
  setPill("err",`error ${e>=0?"+":"−"}${Math.abs(e).toFixed(2)}°`,st>=0,"e");
  setPill("c",`∂T/∂c = ${g.c.toFixed(1)}`,st>=1);
  setPill("vc1",`× ${f2(D.c)}`,st>=1,"m");setPill("vc2",`× ${f2(1-D.c)}`,st>=1,"m");
  setPill("b1",`∂T/∂b₁ = ${f2(D.c)} × ${(f.T1[0]-f.T1[1]).toFixed(0)} = ${g.b1.toFixed(1)}`,st>=2);
  setPill("b2",`∂T/∂b₂ = ${f2(1-D.c)} × ${(f.T1[1]-f.T1[2]).toFixed(0)} = ${g.b2.toFixed(1)}`,st>=2);
  setPill("m11",`× ${f2(g.p[0][0])}`,st>=2,"m");setPill("m21",`× ${f2(g.p[1][0])}`,st>=2,"m");setPill("m22",`× ${f2(g.p[1][1])}`,st>=2,"m");setPill("m32",`× ${f2(g.p[2][0])}`,st>=2,"m");
  setPill("a1",`∂T/∂a₁ = 40 × ${f2(g.g1[0])} = ${g.a1.toFixed(1)}`,st>=3);
  setPill("a2",`∂T/∂a₂ = 40 × (${f2(g.p[1][0])} + ${f2(g.p[1][1])}) = ${g.a2.toFixed(1)}`,st>=3);
  setPill("a3",`∂T/∂a₃ = 40 × ${f2(g.g1[2])} = ${g.a3.toFixed(1)}`,st>=3);
  K3.forEach(k=>{const t=$("#n_"+k);t.textContent=st>=4?`${d[k]>=0?"↻ up":"↺ down"} ${sgn(d[k])}`:"";});
  const flows={out:st>=0,v1o:st>=1,v2o:st>=1,m1v1:st>=2,m2v1:st>=2,m2v2:st>=2,m3v2:st>=2};
  for(const k in DPIPES)$("#pf_"+k).setAttribute("opacity",flows[k]?1:0);
  deepLedger(f,e,g,d);shChart();
  $("#shBack").textContent=st<0?"◀ Backward pass":st>=4?"◀ Backward pass (done)":["◀ Next: layer 3","◀ Next: layer 2","◀ Next: layer 1","◀ Next: the nudges"][st];
}
function deepLedger(f,e,g,d){
  const D=SH.D,st=SH.stage,f2=v=>v.toFixed(2),L=[];
  L.push(`<span class="k">forward</span> layer 1: ${f.T1.map(v=>v.toFixed(1)).join(", ")} · layer 2: ${f.T2.map(v=>v.toFixed(1)).join(", ")} · shower ${f.T.toFixed(2)} °C`);
  L.push(`<span class="k">error</span> T − target = <span class="${e<0?"cold":"hot"}">${e>=0?"+":"−"}${Math.abs(e).toFixed(2)}</span>  <span class="k">loss ½·error² =</span> ${(e*e/2).toFixed(3)}`);
  if(st>=1)L.push(`<span class="k">layer 3</span> ∂T/∂c = ${f.T2[0].toFixed(1)} − ${f.T2[1].toFixed(1)} = ${g.c.toFixed(2)} · blame per degree passed to valve 1: ${f2(D.c)}, valve 2: ${f2(1-D.c)}`);
  if(st>=2)L.push(`<span class="k">layer 2</span> ∂T/∂b₁ = ${g.b1.toFixed(2)} · ∂T/∂b₂ = ${g.b2.toFixed(2)} · blame per degree reaching mixers: ${g.g1.map(f2).join(", ")} <span class="k">(mixer 2 = ${f2(g.p[1][0])} + ${f2(g.p[1][1])}: two paths add)</span>`);
  if(st>=3)L.push(`<span class="k">layer 1</span> ∂T/∂a = 40 × that: ${[g.a1,g.a2,g.a3].map(v=>v.toFixed(1)).join(", ")}`);
  if(st>=4){L.push(`<span class="k">update, all six at once</span> ${K3.map(k=>`Δ${k} = ${sgn(d[k])}`).join(" · ")}`);
    let pred=f.T;for(const k of K3)pred+=g[k]*d[k];L.push(`<span class="k">expected after the step</span> ≈ ${pred.toFixed(2)} °C (${SH.steps} step${SH.steps===1?"":"s"} so far)`);}
  L.push(`<span class="k">blame per degree at each layer's output</span> layer 3: 1.00 · layer 2: ${f2(D.c)}, ${f2(1-D.c)} · layer 1: ${g.g1.map(f2).join(", ")}`);
  $("#shLedger").innerHTML=L.join("<br>");}
function shLedger(f,e,g,d){
  const two=SH.mode===2,st=SH.stage,full=st>=(two?3:2),L=[];
  L.push(`<span class="k">forward</span> ${two?`T_A = 20 + 40·a = ${f.TA.toFixed(2)} · T_B = ${f.TB.toFixed(2)} · `:""}T = ${f.T.toFixed(2)} °C`);
  L.push(`<span class="k">error</span> T − target = <span class="${e<0?"cold":"hot"}">${e>=0?"+":"−"}${Math.abs(e).toFixed(2)}</span>  <span class="k">loss ½·error² =</span> ${(e*e/2).toFixed(3)}`);
  if(st>=0||full){
    if(two){L.push(`<span class="k">valve</span> ∂T/∂b = T_A − T_B = ${g.b.toFixed(2)}`);
      L.push(`<span class="k">blame passed back</span> to A's water × b = ${SH.b.toFixed(2)}, to B's water × (1−b) = ${(1-SH.b).toFixed(2)}`);
      L.push(SH.share?`<span class="k">shared knob</span> ∂T/∂a = ${g.gA.toFixed(1)} (via A) + ${g.gC.toFixed(1)} (via B) = ${g.a.toFixed(1)}`:`<span class="k">mixers</span> ∂T/∂a = ${SH.b.toFixed(2)} × 40 = ${g.a.toFixed(1)} · ∂T/∂c = ${(1-SH.b).toFixed(2)} × 40 = ${g.c.toFixed(1)}`);}
    else L.push(`<span class="k">knob</span> ∂T/∂a = 40`);
    const dd=Object.entries(d).filter(([k])=>!(SH.share&&k==="c")).map(([k,v])=>`Δ${k} = ${sgn(v)}`).join(" · ");
    L.push(`<span class="k">update, all at once</span> Δknob = −lr × error × ∂T/∂knob → ${dd}`);
    let pred=f.T;for(const k in d)pred+=g[k]*d[k];
    L.push(`<span class="k">expected after the step</span> ≈ ${pred.toFixed(2)} °C (${SH.steps} step${SH.steps===1?"":"s"} so far)`);}
  $("#shLedger").innerHTML=L.join("<br>");}
function shChart(){const cv=$("#shChart"),x=cv.getContext("2d"),W=cv.width,H=cv.height;x.clearRect(0,0,W,H);
  const h=SH.hist.length?SH.hist:[shForward().T],n=Math.max(10,h.length),lo=15,hi=65,Y=T=>H-18-(T-lo)/(hi-lo)*(H-28),X=i=>30+i/(n-1)*(W-40);
  x.strokeStyle=CS.grid;x.lineWidth=1;x.font="11px JetBrains Mono, monospace";x.fillStyle=CS.muted;
  [20,40,60].forEach(T=>{x.beginPath();x.moveTo(30,Y(T));x.lineTo(W-10,Y(T));x.stroke();x.fillText(T+"°",2,Y(T)+4);});
  x.strokeStyle=CS.accent;x.setLineDash([5,4]);x.beginPath();x.moveTo(30,Y(SH.target));x.lineTo(W-10,Y(SH.target));x.stroke();x.setLineDash([]);
  x.fillStyle=CS.accent;x.fillText("target",W-54,Y(SH.target)-5);
  x.strokeStyle=CS.fg;x.lineWidth=2;x.beginPath();h.forEach((T,i)=>i?x.lineTo(X(i),Y(T)):x.moveTo(X(i),Y(T)));x.stroke();
  h.forEach((T,i)=>{x.fillStyle=tempCol(T);x.beginPath();x.arc(X(i),Y(T),3.5,0,7);x.fill();});
  x.fillStyle=CS.muted;x.fillText("step →",W-52,H-3);}

/* ---- controls ---- */
function stopAuto(){clearInterval(SH.auto);SH.auto=null;if($("#shAuto"))$("#shAuto").textContent="▶ Auto-train";}
function backward(){const last=SH.mode===3?4:SH.mode===2?3:2;stopAuto();
  if(SH.stage>=last)SH.stage=-1;SH.stage++;drawShower();}
function syncShowerUI(){
  $("#shMode1").setAttribute("aria-pressed",SH.mode===1);$("#shMode2").setAttribute("aria-pressed",SH.mode===2);$("#shMode3").setAttribute("aria-pressed",SH.mode===3);
  $("#shShareWrap").hidden=SH.mode!==2;$("#shShare").checked=SH.share;
  $("#shLR").value=SH.lr;$("#shLRv").textContent=SH.lr.toFixed(4);$("#shTgt").value=SH.target;$("#shTgtv").textContent=SH.target+" °C";
  $$("[data-lr]").forEach(b=>b.setAttribute("aria-pressed",Math.abs(+b.dataset.lr-SH.lr)<1e-9));
  const fac=(()=>{const g=shGrad(shForward()),keys=SH.mode===3?K3:["a","b","c"];return keys.filter(k=>g[k]!=null&&!(SH.share&&k==="c")).reduce((s,k)=>s+g[k]*g[k],0);})()*SH.lr;
  $("#shLRnote").textContent=fac>2?"Too big: each step overshoots by more than it corrects, so the temperature swings back and forth.":fac>1.05?"A bit big: it overshoots, then settles.":fac<.35?"Small: safe, but it takes many steps.":"About right: close to the target in a step or two.";}
function renderShower(){const want=SH.mode===3?"deep":"basic";if(SH.built!==want){want==="deep"?buildDeep():buildShower();if(!SH.hist.length)SH.hist=[shForward().T];}syncShowerUI();drawShower();}
function bindShower(){
  $("#shMode1").onclick=()=>{stopAuto();SH.mode=1;SH.share=false;SH.stage=-1;SH.hist=[shForward().T];SH.steps=0;renderShower();};
  $("#shMode2").onclick=()=>{stopAuto();SH.mode=2;SH.stage=-1;SH.hist=[shForward().T];SH.steps=0;renderShower();};
  $("#shMode3").onclick=()=>{stopAuto();SH.mode=3;SH.share=false;SH.stage=-1;SH.hist=[shForward().T];SH.steps=0;renderShower();};
  $("#shShuffle").onclick=()=>{stopAuto();const r=()=>.05+.9*Math.random();if(SH.mode===3)K3.forEach(k=>SH.D[k]=r());else{SH.a=r();SH.b=r();SH.c=SH.share?SH.a:r();}SH.stage=-1;SH.steps=0;SH.hist=[shForward().T];renderShower();};
  $("#shDeep").onclick=()=>{stopAuto();Object.assign(SH,{mode:3,share:false,target:38,lr:.001,stage:-1,steps:0});SH.D={...DSTART};SH.hist=[shForward().T];renderShower();};
  $("#shVanish").onclick=()=>{stopAuto();Object.assign(SH,{mode:3,share:false,stage:-1,steps:0});Object.assign(SH.D,{c:.12,b1:.15});SH.hist=[shForward().T];renderShower();SH.stage=4;drawShower();};
  $("#shShare").onchange=e=>{stopAuto();SH.share=e.target.checked;if(SH.share)SH.c=SH.a;SH.stage=-1;SH.hist=[shForward().T];SH.steps=0;renderShower();};
  $("#shTgt").oninput=e=>{SH.target=+e.target.value;SH.stage=-1;renderShower();};
  $("#shLR").oninput=e=>{SH.lr=+e.target.value;renderShower();};
  $$("[data-lr]").forEach(b=>b.onclick=()=>{SH.lr=+b.dataset.lr;renderShower();});
  $("#shBack").onclick=backward;
  $("#shStep").onclick=()=>{stopAuto();shStep();renderShower();};
  $("#shAuto").onclick=()=>{if(SH.auto){stopAuto();return;}$("#shAuto").textContent="❚❚ Pause";
    SH.auto=setInterval(()=>{shStep();renderShower();const e=shForward().T-SH.target;if(Math.abs(e)<.02||SH.steps>=60)stopAuto();},reduced?60:450);};
  $("#shReset").onclick=()=>{stopAuto();Object.assign(SH,{a:START.a,c:START.c,b:START.b,target:38,lr:.001,stage:-1,steps:0});SH.D={...DSTART};if(SH.share)SH.c=SH.a;SH.hist=[shForward().T];renderShower();};
  $("#shShut").onclick=()=>{stopAuto();Object.assign(SH,{mode:2,share:false,b:0,stage:-1,steps:0});SH.hist=[shForward().T];renderShower();SH.stage=3;drawShower();};
  $("#shExample").onclick=()=>{stopAuto();Object.assign(SH,{mode:2,share:false,a:.5,c:.25,b:.6,target:38,lr:.001,stage:-1,steps:0});SH.hist=[shForward().T];renderShower();};
}
bindShower();

/* ================= Boot ================= */
if(typeof THREE==="undefined"){const d=document.createElement("div");d.className="glerr";d.textContent="The 3D library didn't load. Check your connection and reload; the panel still works.";stageEl.appendChild(d);}
else initGL();
redrawPad();preset(2);showSections();requestAnimationFrame(loop);
function route(){const h=location.hash,a=h==="#atlas",w=h==="#shower";$("#atlas").hidden=!a;$("#shower").hidden=!w;$(".app").hidden=a||w;
  if(!w)stopAuto();
  if(a){renderAtlas();window.scrollTo(0,0);}else if(w){renderShower();window.scrollTo(0,0);}else{resize();refresh();}}
window.addEventListener("hashchange",route);route();
$$(".jump button").forEach(b=>b.onclick=()=>document.getElementById(b.dataset.j).scrollIntoView({behavior:reduced?"auto":"smooth"}));
window.__S=S;window.__setStage=setStage;window.__preset=preset;window.__net=()=>NET;
})();
