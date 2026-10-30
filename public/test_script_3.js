
function safeGet(id) {
    const el = document.getElementById(id);
    if (el) return el;
    // Return a dummy object that absorbs all assignments and method calls
    const noop = () => {};
    const dummyCtx = { clearRect:noop, moveTo:noop, lineTo:noop, fill:noop, stroke:noop, beginPath:noop, closePath:noop, arc:noop, fillText:noop, save:noop, restore:noop, setLineDash:noop, measureText:()=>({width:0}), scale:noop, translate:noop, setTransform:noop, fillRect:noop };
    const dummy = {
        style: {},
        classList: { add:noop, remove:noop, contains:()=>false, toggle:noop },
        setAttribute: noop,
        getAttribute: () => null,
        addEventListener: noop,
        removeEventListener: noop,
        getContext: () => dummyCtx,
        appendChild: noop,
        removeChild: noop,
        click: noop,
        textContent: '',
        innerHTML: '',
        value: '',
        parentElement: null,
        getBoundingClientRect: () => ({ width: 400, height: 300, top: 0, left: 0, bottom: 300, right: 400 })
    };
    return new Proxy(dummy, {
        set(target, prop, val) { target[prop] = val; return true; },
        get(target, prop) {
            if(prop in target) return target[prop];
            return undefined;
        }
    });
}


/* ============ Engine data (filled from the backend bundle; no synthetic data) ============ */
let data = [];
let N = 0;
let currentCycle = 0;
let selectedEngine = null;
const COMPONENTS = ['fan', 'lpc', 'hpc', 'hpt', 'lpt'];
const COMP_LABEL = { fan: 'Fan', lpc: 'LPC', hpc: 'HPC', hpt: 'HPT', lpt: 'LPT' };
const COMP_SECTION = { fan: 'inlet', lpc: 'compressor', hpc: 'compressor', hpt: 'turbine', lpt: 'turbine' };

/* ============ Helpers ============ */
// Declared early: render() runs during init, before the integration block at the end of the file.
const backend = { online: false, payload: null, engines: [], uploads: {}, explainTimer: null, explainKey: null, currentId: null };
const SEV_COLOR = { warning: '#d97706', alarm: '#ea580c', emergency: '#dc2626', normal: '#059669' };
let bandEdges = [0.75, 0.55, 0.30];   // from the bundle manifest: Normal / Warning / Alarm / Emergency
function bandIdx(v){ return isFinite(v) ? bandEdges.filter(e => v < e).length : 0; }
const BAND_VAR = ['var(--teal)','var(--amber)','var(--orange)','var(--red)'];
const BAND_RAW = ['#6db076','#e5a755','#e07455','#d14f4f'];
const BAND_LABEL = ['Normal','Warning','Alarm','Emergency'];
const bandColor = v => BAND_VAR[bandIdx(v)];
const bandColorRaw = v => BAND_RAW[bandIdx(v)];
const bandLabel = v => BAND_LABEL[bandIdx(v)];
const fmtN = (v, dp) => (v === null || v === undefined || !isFinite(v)) ? '—' : Number(v).toFixed(dp);
const pct = v => (v === null || v === undefined || !isFinite(v)) ? '—' : (v >= 0 ? '+' : '') + Number(v).toFixed(2) + '%';
const bandDesc = {
  'Normal':'Operating well, no special intervention required.',
  'Warning':'Continuous monitoring required to identify potential risks.',
  'Alarm':'Schedule planned maintenance.',
  'Emergency':'Immediately ground the aircraft and perform maintenance.'
};

const fmt = {
  eff: v => v.toFixed(3),                             // Always 3 dp (e.g. 0.885)
  pr:  v => v.toFixed(2),                             // Always 2 dp (e.g. 10.72)
  temp: v => Math.round(v).toLocaleString() + ' K',   // Always 0 dp K (e.g. 647 K)
  press: v => Math.round(v / 1000).toLocaleString() + ' kPa', // Always 0 dp kPa (e.g. 226 kPa)
  hi:  v => v.toFixed(2),                             // Always 2 dp (e.g. 0.83)
  flow: v => v.toFixed(3) + ' kg/s',
  mach: v => v.toFixed(2)
};

function lerpTemp(a,b,f){ return a+(b-a)*f; }

function tempColor(T){
  const stops = [
    {t:280,  c:[0x1e,0x3a,0x8a]},  // Deep blue
    {t:400,  c:[0x03,0x69,0xa1]},  // Blue
    {t:550,  c:[0x08,0x91,0xb2]},  // Cyan
    {t:700,  c:[0x06,0xb6,0xd4]},  // Light cyan
    {t:850,  c:[0x22,0xc5,0x5e]},  // Green
    {t:1000, c:[0xea,0xb3,0x08]},  // Yellow
    {t:1200, c:[0xf9,0x73,0x16]},  // Orange
    {t:1500, c:[0xdc,0x26,0x26]},  // Red
    {t:1800, c:[0xfe,0xf3,0xc7]},  // Light yellow/white
  ];
  let lo = stops[0], hi = stops[stops.length-1];
  if(T<=stops[0].t){ lo=hi=stops[0]; }
  else if(T>=stops[stops.length-1].t){ lo=hi=stops[stops.length-1]; }
  else{
    for(let i=0;i<stops.length-1;i++){
      if(T>=stops[i].t && T<=stops[i+1].t){ lo=stops[i]; hi=stops[i+1]; break; }
    }
  }
  const f = hi.t===lo.t ? 0 : (T-lo.t)/(hi.t-lo.t);
  const r = Math.round(lo.c[0]+(hi.c[0]-lo.c[0])*f);
  const g = Math.round(lo.c[1]+(hi.c[1]-lo.c[1])*f);
  const b = Math.round(lo.c[2]+(hi.c[2]-lo.c[2])*f);
  return (r<<16)|(g<<8)|b;
}

function drawGauge(svgEl, value, sigma = 0.02){
  const cx=65,cy=70,r=54;
  const a0=Math.PI, a1=0;
  const frac = Math.max(0,Math.min(1,value));
  const ang = a0 + (a1-a0)*frac;
  const track = `M ${cx-r} ${cy} A ${r} ${r} 0 0 1 ${cx+r} ${cy}`;
  const x2 = cx + r*Math.cos(ang), y2 = cy + r*Math.sin(ang);
  const large = frac>0.5?1:0;
  const arc = `M ${cx-r} ${cy} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`;
  const nx = cx + (r-14)*Math.cos(ang), ny = cy + (r-14)*Math.sin(ang);

  let uncertPath = '';
  if(sigma && sigma > 0){
    const fLow = Math.max(0, Math.min(1, value - sigma));
    const fHigh = Math.max(0, Math.min(1, value + sigma));
    const angLow = a0 + (a1-a0)*fLow;
    const angHigh = a0 + (a1-a0)*fHigh;
    const xLow = cx + r*Math.cos(angLow), yLow = cy + r*Math.sin(angLow);
    const xHigh = cx + r*Math.cos(angHigh), yHigh = cy + r*Math.sin(angHigh);
    const largeUncert = (fHigh - fLow) > 0.5 ? 1 : 0;
    uncertPath = `<path d="M ${xLow} ${yLow} A ${r} ${r} 0 ${largeUncert} 1 ${xHigh} ${yHigh}" fill="none" stroke="${bandColorRaw(value)}" stroke-width="17" stroke-opacity="0.32" stroke-linecap="round"/>`;
  }

  svgEl.innerHTML = `
    <path d="${track}" fill="none" stroke="#1a222b" stroke-width="9" stroke-linecap="round"/>
    ${uncertPath}
    <path d="${arc}" fill="none" stroke="${bandColorRaw(value)}" stroke-width="9" stroke-linecap="round"/>
    <line x1="${cx}" y1="${cy}" x2="${nx}" y2="${ny}" stroke="#d7dee6" stroke-width="1.5"/>
    <circle cx="${cx}" cy="${cy}" r="3" fill="#d7dee6"/>
  `;
}

function drawSpark(svgEl, arr, color, vb, baselineVal = null){
  const w=vb[0], h=vb[1];
  let min=Math.min(...arr), max=Math.max(...arr);
  if(baselineVal !== null){
    min = Math.min(min, baselineVal - 0.05);
    max = Math.max(max, baselineVal + 0.02);
  }
  const pts = arr.map((v,i)=>{
    const x = arr.length > 1 ? i/(arr.length-1)*w : 0;   // single point at cycle 0
    const y = h - ((v-min)/((max-min)||1))*h*0.80 - h*0.10;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');

  let baseSvg = '';
  if(baselineVal !== null){
    const yBase = h - ((baselineVal-min)/((max-min)||1))*h*0.80 - h*0.10;
    const polyPts = `0,${yBase.toFixed(1)} ${pts} ${w},${yBase.toFixed(1)}`;
    baseSvg = `
      <polygon points="${polyPts}" fill="${color}" fill-opacity="0.10"/>
      <line x1="0" y1="${yBase.toFixed(1)}" x2="${w}" y2="${yBase.toFixed(1)}" stroke="#4a5561" stroke-width="1" stroke-dasharray="2,2" opacity="0.8"/>
      <text x="2" y="${Math.max(8, yBase - 2).toFixed(1)}" font-size="7.5" fill="#7c8896" font-family="IBM Plex Mono">BASE ${baselineVal.toFixed(2)}</text>
    `;
  }

  svgEl.setAttribute('viewBox',`0 0 ${w} ${h}`);
  svgEl.innerHTML = `
    ${baseSvg}
    <polyline points="${pts}" fill="none" stroke="${color}" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/>
  `;
}

function drawTrend(svgEl, upTo){
  if (!data.length) return;
  const w=1200,h=170, padL=42, padR=44, padT=12, padB=22;
  const iw=w-padL-padR, ih=h-padT-padB;
  const slice = data.slice(0, upTo+1);
  const hiArr = slice.map(d=>d.ohi);
  const rulArr = slice.map(d=>d.rul);
  const rulMax = Math.max(...data.map(d=>d.rul));

  const xAt = i => padL + i/(N-1)*iw;
  const yHi = v => padT + (1-v)*ih;
  const yRul = v => padT + (1-v/rulMax)*ih;

  const hiPts = hiArr.map((v,i)=>`${xAt(i).toFixed(1)},${yHi(v).toFixed(1)}`).join(' ');
  const rulPts = rulArr.map((v,i)=>`${xAt(i).toFixed(1)},${yRul(v).toFixed(1)}`).join(' ');

  const bands = [
    {y0:0,y1:0.30,c:'rgba(220,38,38,0.08)'},
    {y0:0.30,y1:0.55,c:'rgba(234,88,12,0.07)'},
    {y0:0.55,y1:0.75,c:'rgba(245,158,11,0.06)'},
    {y0:0.75,y1:1,c:'rgba(79,209,181,0.05)'},
  ];
  let bandRects = bands.map(b=>{
    const y1 = padT+(1-b.y1)*ih, y0 = padT+(1-b.y0)*ih;
    return `<rect x="${padL}" y="${y1}" width="${iw}" height="${y0-y1}" fill="${b.c}"/>`;
  }).join('');

  let gridLines = '';
  for(let g=0; g<=4; g++){
    const y = padT + g/4*ih;
    const hiVal = (1 - g/4).toFixed(2);
    const rulVal = Math.round((1 - g/4) * rulMax);
    gridLines += `<line x1="${padL}" y1="${y}" x2="${w-padR}" y2="${y}" stroke="#1a222b" stroke-width="1"/>`;
    gridLines += `<text x="${padL-6}" y="${y+3}" text-anchor="end" font-size="9" fill="#7c8896" font-family="IBM Plex Mono">${hiVal}</text>`;
    gridLines += `<text x="${w-padR+6}" y="${y+3}" text-anchor="start" font-size="9" fill="#8b95a1" font-family="IBM Plex Mono">${rulVal}</text>`;
  }

  const markerX = xAt(upTo);

  svgEl.setAttribute('viewBox', `0 0 ${w} ${h}`);
  svgEl.innerHTML = `
    ${bandRects}
    ${gridLines}
    <polyline points="${rulPts}" fill="none" stroke="#8b95a1" stroke-width="1.4" stroke-dasharray="4,3"/>
    <polyline points="${hiPts}" fill="none" stroke="#5fb3d9" stroke-width="2"/>
    <line x1="${markerX}" y1="${padT}" x2="${markerX}" y2="${padT+ih}" stroke="#d7dee6" stroke-width="1" stroke-dasharray="2,2" opacity="0.5"/>
    <circle cx="${markerX}" cy="${yHi(hiArr[hiArr.length-1])}" r="3.5" fill="#5fb3d9"/>
    <text x="${padL}" y="${h-5}" font-size="9" fill="#7c8896" font-family="IBM Plex Mono">CYCLE 0</text>
    <text x="${w-padR}" y="${h-5}" text-anchor="end" font-size="9" fill="#7c8896" font-family="IBM Plex Mono">CYCLE ${N-1}</text>
  `;
}

/* ============ THREE.JS 3D ENGINE MODEL ============ */
let scene, camera, renderer, engineGroup, compressorGroup, combustorMesh, turbineGroup, shaftMesh, spoolGroup;
let inletGroup, combustorGroup, nozzleGroup;
let componentGroupsMap = {}, clickableGroups = [];
let rotY = -0.3, rotX = -0.15, camDist = 2100.0;
let targetPanOffset = 0, currentPanOffset = 0;
let isDragging=false, lastX=0, lastY=0, autoRotate=false;
let raycaster = new THREE.Raycaster(), mouseVec = new THREE.Vector2();
let selectedComponent = null, boxHelper = null;
let touchStartPos = null, touchMoved = false;

function makeMat(color){
  return new THREE.MeshStandardMaterial({ color, metalness:0.65, roughness:0.35, emissive:0x000000, emissiveIntensity:0.4 });
}

function initThree(){
  const canvas = safeGet('engineCanvas');
  const frame = canvas.parentElement;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(32, frame.clientWidth/frame.clientHeight, 200, 100000);
  renderer = new THREE.WebGLRenderer({canvas, antialias:true, alpha:true, localClippingEnabled:true, precision:'highp'}); renderer.setClearColor(0x000000, 0);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(frame.clientWidth, frame.clientHeight);
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.shadowMap.enabled = false; // shadows disabled — per-frame shadow map rebuild was the flicker source

  scene.add(new THREE.AmbientLight(0xffffff, 0.45)); // Reduced from 0.95 to 0.45 to make shadows more visible
  
  const key = new THREE.DirectionalLight(0xffffff, 1.8); 
  key.position.set(1000, 1500, 1200); // Scaled positions to ensure lights are positioned outside the engine volume
  key.castShadow = true;
  key.shadow.mapSize.width = 4096; // 4K maps for crisp shadow profiles
  key.shadow.mapSize.height = 4096;
  key.shadow.camera.near = 10;
  key.shadow.camera.far = 6000;
  key.shadow.camera.left = -1000;
  key.shadow.camera.right = 1000;
  key.shadow.camera.top = 1000;
  key.shadow.camera.bottom = -1000;
  key.shadow.bias = -0.0002;
  scene.add(key);
  
  const rim = new THREE.DirectionalLight(0x7fb3ff, 1.2); 
  rim.position.set(-1000, 500, -1000);
  rim.castShadow = true;
  rim.shadow.mapSize.width = 1024;
  rim.shadow.mapSize.height = 1024;
  rim.shadow.camera.near = 10;
  rim.shadow.camera.far = 6000;
  rim.shadow.camera.left = -1000;
  rim.shadow.camera.right = 1000;
  rim.shadow.camera.top = 1000;
  rim.shadow.camera.bottom = -1000;
  scene.add(rim);
  
  const fill = new THREE.DirectionalLight(0xffcc99, 0.8); 
  fill.position.set(1000, -1000, 500);
  scene.add(fill);
  
  const top = new THREE.DirectionalLight(0xffffff, 0.8); 
  top.position.set(0, 1500, 0);
  top.castShadow = true;
  top.shadow.camera.left = -1000;
  top.shadow.camera.right = 1000;
  top.shadow.camera.top = 1000;
  top.shadow.camera.bottom = -1000;
  scene.add(top);
  
  const detail = new THREE.DirectionalLight(0xccddff, 0.6);
  detail.position.set(-1000, 800, 800);
  scene.add(detail);

  engineGroup = new THREE.Group();
  scene.add(engineGroup);
  loadEngineAsset();
  updateCamera();

  let mouseDownPos = null, mouseMoved = false;
  canvas.addEventListener('mousedown', e=>{
    isDragging=true; autoRotate=false; lastX=e.clientX; lastY=e.clientY;
    mouseDownPos = {x:e.clientX,y:e.clientY}; mouseMoved = false;
  });
  window.addEventListener('mouseup', e=>{
    isDragging=false;
    if(mouseDownPos && !mouseMoved){
      const key = getIntersectComponent(e.clientX, e.clientY, canvas);
      if(key) selectComponent(key);
    }
    mouseDownPos = null;
  });
  window.addEventListener('mousemove', e=>{
    if(mouseDownPos && Math.hypot(e.clientX-mouseDownPos.x, e.clientY-mouseDownPos.y) > 4) mouseMoved = true;
    if(!isDragging){
      const key = getIntersectComponent(e.clientX, e.clientY, canvas);
      canvas.style.cursor = key ? 'pointer' : 'grab';
      return;
    }
    const dx = e.clientX-lastX, dy = e.clientY-lastY;
    rotY += dx*0.008; rotX = Math.max(-1.1, Math.min(1.1, rotX + dy*0.008));
    lastX=e.clientX; lastY=e.clientY;
    updateCamera();
  });
  canvas.addEventListener('touchstart', e=>{
    isDragging=true; autoRotate=false; lastX=e.touches[0].clientX; lastY=e.touches[0].clientY;
    touchStartPos = {x:lastX, y:lastY}; touchMoved = false;
  });
  canvas.addEventListener('touchmove', e=>{
    if(!isDragging) return;
    if(touchStartPos && Math.hypot(e.touches[0].clientX-touchStartPos.x, e.touches[0].clientY-touchStartPos.y) > 4) touchMoved = true;
    const dx=e.touches[0].clientX-lastX, dy=e.touches[0].clientY-lastY;
    rotY += dx*0.008; rotX = Math.max(-1.1, Math.min(1.1, rotX + dy*0.008));
    lastX=e.touches[0].clientX; lastY=e.touches[0].clientY;
    updateCamera();
  });
  window.addEventListener('touchend', e=>{
    isDragging=false;
    if(touchStartPos && !touchMoved && e.changedTouches && e.changedTouches[0]){
      const key = getIntersectComponent(e.changedTouches[0].clientX, e.changedTouches[0].clientY, canvas);
      if(key) selectComponent(key);
    }
    touchStartPos = null;
  });
  canvas.addEventListener('wheel', e=>{
    e.preventDefault();
    camDist = Math.max(300, Math.min(6000, camDist + e.deltaY * 1.5));
    updateCamera();
  }, {passive:false});

  window.addEventListener('resize', ()=>{
    const f = canvas.parentElement;
    camera.aspect = f.clientWidth/f.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(f.clientWidth, f.clientHeight);
  });
}

function updateCamera(){
  camera.position.set(
    camDist*Math.cos(rotX)*Math.sin(rotY),
    camDist*Math.sin(rotX),
    camDist*Math.cos(rotX)*Math.cos(rotY)
  );
  camera.lookAt(0,0,0);
  camera.updateMatrix();
  const panVec = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0).multiplyScalar(currentPanOffset);
  camera.position.add(panVec);
  camera.lookAt(panVec);
}

function makeBladeRing(x, r, count, group){
  for(let k=0;k<count;k++){
    const ang = (k/count)*Math.PI*2;
    const blade = new THREE.Mesh(new THREE.BoxGeometry(0.30, 0.02, 0.13), makeMat(0x5a6472));
    blade.position.set(x, Math.cos(ang)*(r+0.09), Math.sin(ang)*(r+0.09));
    blade.rotation.x = ang;
    blade.rotation.z = 0.36;
    blade.userData.isRotor = true;
    blade.userData.isBlade = true;
    group.add(blade);
  }
}

// Vertex coloring state — shared so updateEngineVisuals can reach it
let gltfMeshes = []; // {colAttr, zoneIndices} per mesh

// Utility to convert Base64 string directly to ArrayBuffer in memory
function base64ToArrayBuffer(base64) {
  var binary_string = window.atob(base64);
  var len = binary_string.length;
  var bytes = new Uint8Array(len);
  for (var i = 0; i < len; i++) {
    bytes[i] = binary_string.charCodeAt(i);
  }
  return bytes.buffer;
}

// Override THREE.FileLoader.prototype.load to handle data: URIs synchronously without XHR/fetch file:// CORS issues
if (typeof THREE !== 'undefined' && THREE.FileLoader) {
  const originalFileLoaderLoad = THREE.FileLoader.prototype.load;
  THREE.FileLoader.prototype.load = function (url, onLoad, onProgress, onError) {
    if (typeof url === 'string' && url.startsWith('data:')) {
      try {
        const base64Index = url.indexOf('base64,');
        if (base64Index !== -1) {
          const base64Data = url.substring(base64Index + 7);
          const buffer = base64ToArrayBuffer(base64Data);
          if (onLoad) setTimeout(() => onLoad(buffer), 0);
          return;
        }
      } catch (e) {
        console.warn('Error decoding base64 Data URI in FileLoader override:', e);
      }
    }
    return originalFileLoaderLoad.call(this, url, onLoad, onProgress, onError);
  };
}

function loadEngineAsset(){
  const inlineScript = document.getElementById('engine-glb-data');
  const loader = new THREE.GLTFLoader();

  const handleLoadedGLTF = (gltf) => {
    if (engineGroup.children.length) engineGroup.clear();
    gltfMeshes = [];

    const model = gltf.scene;
    model.name = "gltfRoot";
    model.scale.set(1.5, 1.5, 1.5);
    
    // We will wrap the model and the click zones in a container so they move together
    const alignmentGroup = new THREE.Group();
    alignmentGroup.add(model);

    // Simple phong + vertex colors — no custom shader, no onBeforeCompile, no surprises
    const baseMat = new THREE.MeshPhongMaterial({
      vertexColors: true,
      specular: new THREE.Color(0x444444),
      shininess: 30,
      side: THREE.DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1
    });

    model.traverse((obj) => {
      if (obj.isMesh) {
        obj.castShadow = false;
        obj.receiveShadow = false;
        // Synchronously pre-allocate color buffer filled with neutral steel grey
        const geo = obj.geometry;
        geo.computeVertexNormals();
        const count = geo.attributes.position.count;
        const colArr = new Float32Array(count * 3);
        for (let i = 0; i < count * 3; i += 3) {
          colArr[i] = 0.54; colArr[i+1] = 0.57; colArr[i+2] = 0.62;
        }
        geo.setAttribute('color', new THREE.BufferAttribute(colArr, 3));
        obj.material = baseMat;
        gltfMeshes.push({ obj, geo, colArr, zoneIndices: null });
      }
    });

    engineGroup.add(alignmentGroup);
    renderer.clippingPlanes = []; // no global clipping — was causing flicker
    
    // Default zones as fallback
    let zonesData = {
      inlet:      { center: [320, -13, -4], size: [203, 375, 375] },
      compressor: { center: [75, -14, 2], size: [287, 375, 385] },
      combustor:  { center: [-202, -17, -12], size: [251, 470, 566] },
      turbine:    { center: [-475, -54, -4], size: [286, 616, 592] },
      nozzle:     { center: [-755, -34, 2], size: [286, 647, 594] }
    };

    // Helper to map labels key from engine_labels.json to component keys
    function mapKeyToComponent(labelKey) {
      const key = labelKey.toLowerCase();
      if (key.includes('inlet') || key.includes('intake')) return 'inlet';
      if (key.includes('compressor') || key.includes('fan')) return 'compressor';
      if (key.includes('combustor') || key.includes('combustion')) return 'combustor';
      if (key.includes('turbine')) return 'turbine';
      if (key.includes('nozzle') || key.includes('exhaust')) return 'nozzle';
      return null;
    }

    // Synchronously parse embedded engine labels data
    try {
      const labelsScript = document.getElementById('engine-labels-data');
      if (labelsScript && labelsScript.textContent && labelsScript.textContent.trim()) {
        const labels = JSON.parse(labelsScript.textContent);
        labels.forEach(item => {
          const compKey = mapKeyToComponent(item.key);
          if (compKey) {
            zonesData[compKey] = {
              center: item.center,
              size: item.size
            };
          }
        });
      }
    } catch(e) {
      console.warn('[WARNING] Error parsing inline labels data:', e);
    }
    buildClickZones();

    function buildClickZones() {
      // Pre-compute which zone each vertex belongs to (done ONCE, not per-frame)
      const compKeys = ['inlet', 'compressor', 'combustor', 'turbine', 'nozzle'];
      const _v = new THREE.Vector3();
      gltfMeshes.forEach(entry => {
        entry.obj.updateMatrixWorld(true);
        const posAttr = entry.geo.attributes.position;
        const count = posAttr.count;
        const zoneIdx = new Uint8Array(count).fill(255);
        for (let i = 0; i < count; i++) {
          _v.fromBufferAttribute(posAttr, i);
          entry.obj.localToWorld(_v);
          for (let j = 0; j < compKeys.length; j++) {
            const z = zonesData[compKeys[j]];
            if (!z) continue;
            if (Math.abs(_v.x - z.center[0]) <= z.size[0]/2 &&
                Math.abs(_v.y - z.center[1]) <= z.size[1]/2 &&
                Math.abs(_v.z - z.center[2]) <= z.size[2]/2) {
              zoneIdx[i] = j; break;
            }
          }
        }
        entry.zoneIndices = zoneIdx;
      });
      // Apply initial colours immediately
      updateEngineVisuals(data[currentCycle]);

      const makeZoneMat = () => new THREE.MeshBasicMaterial({
        color: 0x000000, 
        transparent: true, 
        opacity: 0.0, 
        depthWrite: false,
        colorWrite: false,
        side: THREE.DoubleSide
      });
      
      const inletZone = new THREE.Mesh(new THREE.BoxGeometry(...zonesData.inlet.size), makeZoneMat());
      inletZone.position.set(...zonesData.inlet.center);
      inletZone.userData.componentKey = 'inlet';
      inletZone.raycast = THREE.Mesh.prototype.raycast;
      alignmentGroup.add(inletZone);
      
      const compressorZone = new THREE.Mesh(new THREE.BoxGeometry(...zonesData.compressor.size), makeZoneMat());
      compressorZone.position.set(...zonesData.compressor.center);
      compressorZone.userData.componentKey = 'compressor';
      compressorZone.raycast = THREE.Mesh.prototype.raycast;
      alignmentGroup.add(compressorZone);
      
      const combustorZone = new THREE.Mesh(new THREE.BoxGeometry(...zonesData.combustor.size), makeZoneMat());
      combustorZone.position.set(...zonesData.combustor.center);
      combustorZone.userData.componentKey = 'combustor';
      combustorZone.raycast = THREE.Mesh.prototype.raycast;
      alignmentGroup.add(combustorZone);
      
      const turbineZone = new THREE.Mesh(new THREE.BoxGeometry(...zonesData.turbine.size), makeZoneMat());
      turbineZone.position.set(...zonesData.turbine.center);
      turbineZone.userData.componentKey = 'turbine';
      turbineZone.raycast = THREE.Mesh.prototype.raycast;
      alignmentGroup.add(turbineZone);
      
      const nozzleZone = new THREE.Mesh(new THREE.BoxGeometry(...zonesData.nozzle.size), makeZoneMat());
      nozzleZone.position.set(...zonesData.nozzle.center);
      nozzleZone.userData.componentKey = 'nozzle';
      nozzleZone.raycast = THREE.Mesh.prototype.raycast;
      alignmentGroup.add(nozzleZone);
      
      clickableGroups = [inletZone, compressorZone, combustorZone, turbineZone, nozzleZone];
      componentGroupsMap = {
        inlet: inletZone,
        compressor: compressorZone,
        combustor: combustorZone,
        turbine: turbineZone,
        nozzle: nozzleZone
      };

      // Now that zones are built against the unrotated model, apply the aesthetic transform
      alignmentGroup.rotation.x = 0.2;
      alignmentGroup.position.set(217.5, 34, 0);
      alignmentGroup.updateMatrixWorld(true);

      // Dynamically update HUD anchors to match transformed bounding box tops
      for (const key in hudAnchorsGLTF) {
        const zone = zonesData[key];
        if (zone) {
          const anchor = new THREE.Vector3(
            zone.center[0],
            zone.center[1] + (zone.size[1] / 2) * 0.9,
            zone.center[2]
          );
          anchor.applyMatrix4(alignmentGroup.matrixWorld);
          hudAnchorsGLTF[key].copy(anchor);
        }
      }
      
      camDist = 2100.0;
      updateCamera();
      updateEngineVisuals(data[currentCycle]);
    }
  };

  const hideLoading = () => {
    const loaderEl = document.getElementById('engine-loading-indicator');
    if (loaderEl) loaderEl.style.display = 'none';
  };

  const originalHandleLoaded = handleLoadedGLTF;
  const wrappedHandleLoaded = (gltf) => {
    originalHandleLoaded(gltf);
    hideLoading();
  };

  if (inlineScript && inlineScript.textContent && inlineScript.textContent.trim()) {
    try {
      const glbBuffer = base64ToArrayBuffer(inlineScript.textContent.trim());
      loader.parse(glbBuffer, '', wrappedHandleLoaded, (err) => {
        console.warn('Failed parsing inline GLB ArrayBuffer, trying fallback:', err);
        fallbackFetch();
      });
      return;
    } catch(e) {
      console.warn('Inline GLB parse error:', e);
    }
  }

  function fallbackFetch() {
    loader.load('./models/Turbojet Engine.glb', wrappedHandleLoaded, undefined, (error) => {
      console.warn('GLB load failed, trying GLTF:', error);
      loader.load('./models/Turbojet Engine.gltf', wrappedHandleLoaded, undefined, (err2) => {
        console.warn('GLTF model could not be loaded; falling back to synthetic viewer.', err2);
        buildEngine();
        hideLoading();
        camDist = 10.0;
        updateCamera();
      });
    });
  }

  fallbackFetch();
}

function buildEngine(){
  const neutral = 0x5a6472;
  const darkRim = 0x1c232b;

  // central shaft (spins fast — visual "engine running" cue)
  spoolGroup = new THREE.Group();
  const shaftGeo = new THREE.CylinderGeometry(0.07,0.07,7.0,16);
  shaftMesh = new THREE.Mesh(shaftGeo, makeMat(0x8b95a1));
  shaftMesh.rotation.z = Math.PI/2;
  spoolGroup.add(shaftMesh);
  engineGroup.add(spoolGroup);

  // ================= INLET =================
  inletGroup = new THREE.Group();
  inletGroup.userData.componentKey = 'inlet';

  const spinner = new THREE.Mesh(new THREE.ConeGeometry(0.22,0.55,28), makeMat(0x8b95a1));
  spinner.rotation.z = -Math.PI/2;
  spinner.position.x = -3.62;
  spinner.userData.isBlade = true;
  inletGroup.add(spinner);

  const inletHull = new THREE.Mesh(new THREE.CylinderGeometry(0.80,0.84,0.85,40,1,true), makeMat(neutral));
  inletHull.material.side = THREE.DoubleSide;
  inletHull.rotation.z = Math.PI/2;
  inletHull.position.x = -2.94;
  inletHull.userData.isHull = true;
  inletGroup.add(inletHull);

  const inletLip = new THREE.Mesh(new THREE.TorusGeometry(0.84,0.028,10,40), makeMat(0x8b95a1));
  inletLip.rotation.y = Math.PI/2;
  inletLip.position.x = -3.36;
  inletGroup.add(inletLip);

  for(let k=0;k<8;k++){
    const ang = (k/8)*Math.PI*2;
    const vane = new THREE.Mesh(new THREE.BoxGeometry(0.4,0.02,0.5), makeMat(0x3a4552));
    vane.position.set(-2.95, Math.cos(ang)*0.5, Math.sin(ang)*0.5);
    vane.rotation.x = ang;
    inletGroup.add(vane);
  }
  engineGroup.add(inletGroup);

  // ================= COMPRESSOR — 5 contiguous tapering stages =================
  compressorGroup = new THREE.Group();
  compressorGroup.userData.componentKey = 'compressor';
  const compStart=-2.52, compEnd=-0.72, stages=5, compR0=0.80, compR1=0.50;
  for(let i=0;i<stages;i++){
    const t0=i/stages, t1=(i+1)/stages;
    const r0=compR0-(compR0-compR1)*t0, r1=compR0-(compR0-compR1)*t1;
    const len=(compEnd-compStart)/stages;
    const x0=compStart+len*i, xm=x0+len/2;

    const hull = new THREE.Mesh(new THREE.CylinderGeometry(r1,r0,len,32), makeMat(neutral));
    hull.rotation.z = Math.PI/2; hull.position.x = xm;
    hull.userData.isHull = true; hull.userData.tFrac = (t0+t1)/2;
    compressorGroup.add(hull);

    const rim = new THREE.Mesh(new THREE.CylinderGeometry(r0*1.04,r0*1.04,0.035,32), makeMat(darkRim));
    rim.rotation.z = Math.PI/2; rim.position.x = x0;
    rim.userData.isRotor = true; rim.userData.isBlade = true;
    compressorGroup.add(rim);

    makeBladeRing(x0+len*0.28, r0*0.94, 14, compressorGroup);
  }
  const compExitRim = new THREE.Mesh(new THREE.CylinderGeometry(compR1*1.04,compR1*1.04,0.035,32), makeMat(darkRim));
  compExitRim.rotation.z = Math.PI/2; compExitRim.position.x = compEnd;
  compExitRim.userData.isRotor = true; compExitRim.userData.isBlade = true;
  compressorGroup.add(compExitRim);
  engineGroup.add(compressorGroup);

  // ================= COMBUSTOR =================
  combustorGroup = new THREE.Group();
  combustorGroup.userData.componentKey = 'combustor';

  combustorMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.58,0.52,0.95,40), makeMat(neutral));
  combustorMesh.rotation.z = Math.PI/2; combustorMesh.position.x = -0.245;
  combustorMesh.userData.isHull = true;
  combustorGroup.add(combustorMesh);

  const flame = new THREE.Mesh(new THREE.CylinderGeometry(0.30,0.28,0.7,24), makeMat(0xff5a3c));
  flame.rotation.z = Math.PI/2; flame.position.x = -0.245;
  flame.userData.isFlame = true;
  combustorGroup.add(flame);

  [-0.6, 0.1].forEach(xp=>{
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.585,0.022,8,40), makeMat(0x2a323d));
    ring.rotation.y = Math.PI/2; ring.position.x = xp;
    combustorGroup.add(ring);
  });
  engineGroup.add(combustorGroup);

  // ================= TURBINE — 2 contiguous tapering stages =================
  turbineGroup = new THREE.Group();
  turbineGroup.userData.componentKey = 'turbine';
  const turbStart=0.30, turbEnd=1.05, tstages=2, turbR0=0.50, turbR1=0.40;
  for(let i=0;i<tstages;i++){
    const t0=i/tstages, t1=(i+1)/tstages;
    const r0=turbR0-(turbR0-turbR1)*t0, r1=turbR0-(turbR0-turbR1)*t1;
    const len=(turbEnd-turbStart)/tstages;
    const x0=turbStart+len*i, xm=x0+len/2;

    const hull = new THREE.Mesh(new THREE.CylinderGeometry(r1,r0,len,32), makeMat(neutral));
    hull.rotation.z = Math.PI/2; hull.position.x = xm;
    hull.userData.isHull = true; hull.userData.tFrac = (t0+t1)/2;
    turbineGroup.add(hull);

    const rim = new THREE.Mesh(new THREE.CylinderGeometry(r0*1.05,r0*1.05,0.032,32), makeMat(darkRim));
    rim.rotation.z = Math.PI/2; rim.position.x = x0;
    rim.userData.isRotor = true; rim.userData.isBlade = true;
    turbineGroup.add(rim);

    makeBladeRing(x0+len*0.3, r0*0.95, 16, turbineGroup);
  }
  const turbExitRim = new THREE.Mesh(new THREE.CylinderGeometry(turbR1*1.05,turbR1*1.05,0.032,32), makeMat(darkRim));
  turbExitRim.rotation.z = Math.PI/2; turbExitRim.position.x = turbEnd;
  turbExitRim.userData.isRotor = true; turbExitRim.userData.isBlade = true;
  turbineGroup.add(turbExitRim);
  engineGroup.add(turbineGroup);

  // ================= NOZZLE =================
  nozzleGroup = new THREE.Group();
  nozzleGroup.userData.componentKey = 'nozzle';

  const nozzle = new THREE.Mesh(new THREE.CylinderGeometry(0.20,0.40,1.15,40,1,true), makeMat(neutral));
  nozzle.material.side = THREE.DoubleSide;
  nozzle.rotation.z = Math.PI/2; nozzle.position.x = 1.60;
  nozzle.userData.isHull = true;
  nozzleGroup.add(nozzle);

  const plume = new THREE.Mesh(new THREE.CylinderGeometry(0.02,0.20,0.9,24,1,true),
    new THREE.MeshBasicMaterial({color:0xff5a3c, transparent:true, opacity:0.35, side:THREE.DoubleSide, blending:THREE.AdditiveBlending, depthWrite:false}));
  plume.rotation.z = Math.PI/2; plume.position.x = 2.55;
  plume.userData.isPlume = true;
  nozzleGroup.add(plume);
  engineGroup.add(nozzleGroup);

  // ================= OUTER NACELLE — decorative x-ray casing, not clickable =================
  const nacelle = new THREE.Mesh(new THREE.CylinderGeometry(0.95,0.95,5.2,40,1,true),
    new THREE.MeshBasicMaterial({color:0x5fb3d9, transparent:true, opacity:0.055, side:THREE.DoubleSide, depthWrite:false}));
  nacelle.rotation.z = Math.PI/2; nacelle.position.x = -0.6;
  engineGroup.add(nacelle);

  const nacelleLip = new THREE.Mesh(new THREE.TorusGeometry(0.95,0.02,10,48), makeMat(0x5fb3d9));
  nacelleLip.material.transparent = true; nacelleLip.material.opacity = 0.35;
  nacelleLip.rotation.y = Math.PI/2; nacelleLip.position.x = -3.2;
  engineGroup.add(nacelleLip);

  const pylon = new THREE.Mesh(new THREE.BoxGeometry(0.9,0.55,0.12), makeMat(0x2a323d));
  pylon.position.set(-0.2, 1.15, 0);
  engineGroup.add(pylon);

  clickableGroups = [inletGroup, compressorGroup, combustorGroup, turbineGroup, nozzleGroup];
  componentGroupsMap = {inlet:inletGroup, compressor:compressorGroup, combustor:combustorGroup, turbine:turbineGroup, nozzle:nozzleGroup};
}

/* ---- thermal + health tinting ---- */
function applyThermal(mesh, T, opts={}){
  const hex = tempColor(T);
  const c = new THREE.Color(hex);
  mesh.material.color.copy(c);
  if(mesh.material.emissive){
    const heat = Math.max(0, Math.min(1, (T-700)/1000));
    mesh.material.emissive.copy(c);
    mesh.material.emissiveIntensity = (opts.baseGlow ?? 0.15) + heat*(opts.maxGlow ?? 0.55);
  }
}
function tintRotor(mesh, hex){
  const c = new THREE.Color(hex);
  mesh.material.color.copy(c);
  const critical = (hex === '#e2564f' || hex === '#e07b3f');
  mesh.material.emissive.copy(critical ? c : new THREE.Color(0x000000));
  mesh.material.emissiveIntensity = critical ? 0.4 : 0.05;
}
function updateEngineVisuals(d){
  if (!d) return;
  const T3 = isFinite(d.t3) ? d.t3 : d.t48;            // burner exit: virtual T40 if available
  // Paint vertex colours by station temperature (K): inlet T2, compressor T30, burner T40, turbine T48, nozzle T50
  if (gltfMeshes.length > 0) {
    const zoneColors = [d.ttamb, d.t2, T3, d.t48, d.t4].map(T => new THREE.Color(tempColor(isFinite(T) ? T : 300)));
    const defCol = new THREE.Color(0x8a929e);
    gltfMeshes.forEach(({ geo, zoneIndices }) => {
      if (!zoneIndices) return;
      const colAttr = geo.attributes.color;
      for (let i = 0; i < zoneIndices.length; i++) {
        const z = zoneIndices[i];
        const c = z === 255 ? defCol : zoneColors[z];
        colAttr.setXYZ(i, c.r, c.g, c.b);
      }
      colAttr.needsUpdate = true;
    });
  }

  // Synthetic viewer fallbacks (only runs if the GLB failed to load)
  if (compressorGroup && combustorGroup && turbineGroup && nozzleGroup && compressorGroup.type === 'Group') {
    compressorGroup.children.forEach(m=>{
      if(m.userData.isHull) applyThermal(m, lerpTemp(d.ttamb, d.t2, m.userData.tFrac ?? 1), {baseGlow:0.05, maxGlow:0.22});
      if(m.userData.isRotor) tintRotor(m, bandColorRaw(d.hi_comp_min));
    });
    combustorGroup.children.forEach(m=>{
      if(m.userData.isHull) applyThermal(m, (d.t2+T3)/2, {baseGlow:0.18, maxGlow:0.5});
      if(m.userData.isFlame) applyThermal(m, T3, {baseGlow:0.45, maxGlow:0.95});
    });
    turbineGroup.children.forEach(m=>{
      if(m.userData.isHull) applyThermal(m, lerpTemp(T3, d.t4, m.userData.tFrac ?? 0), {baseGlow:0.18, maxGlow:0.55});
      if(m.userData.isRotor) tintRotor(m, bandColorRaw(d.hi_turb_min));
    });
    nozzleGroup.children.forEach(m=>{
      if(m.userData.isHull) applyThermal(m, d.t4, {baseGlow:0.12, maxGlow:0.4});
      if(m.userData.isPlume) applyThermal(m, d.t4*0.75, {baseGlow:0.3, maxGlow:0.6});
    });
  }

  safeGet('legendFan').style.background = bandColorRaw(d.hi_fan);
  safeGet('legendComp').style.background = bandColorRaw(d.hi_comp_min);
  safeGet('legendTurb').style.background = bandColorRaw(d.hi_turb_min);
}

/* ---- click-to-inspect ---- */
function getIntersectComponent(clientX, clientY, canvas){
  const rect = canvas.getBoundingClientRect();
  mouseVec.x = ((clientX-rect.left)/rect.width)*2-1;
  mouseVec.y = -((clientY-rect.top)/rect.height)*2+1;
  raycaster.setFromCamera(mouseVec, camera);
  const hits = raycaster.intersectObjects(clickableGroups, true);
  if(hits.length===0) return null;
  let obj = hits[0].object;
  while(obj && !obj.userData.componentKey){ obj = obj.parent; }
  return obj ? obj.userData.componentKey : null;
}

function highlightGroup(group){
  if(boxHelper){ scene.remove(boxHelper); boxHelper=null; }
  if(!group) return;
  boxHelper = new THREE.BoxHelper(group, 0xf5d442);
  boxHelper.material.transparent = false;
  boxHelper.material.opacity = 1.0;
  boxHelper.material.depthWrite = false;
  scene.add(boxHelper);
}

function selectComponent(key){
  selectedComponent = key;
  highlightGroup(componentGroupsMap[key]);
  refreshComponentCard();
  safeGet('compCard').classList.add('show');
  const w = window.innerWidth;
  if(w > 768) targetPanOffset = -camDist * 0.14; // Pan camera left by 14% of dist to shift model right without clipping
}
function deselectComponent(){
  selectedComponent = null;
  highlightGroup(null);
  safeGet('compCard').classList.remove('show');
  targetPanOffset = 0;
}

const compInfo = {
  inlet:      { title:'Fan (inlet section)',   hiKey:'hi_fan' },
  compressor: { title:'Compressor: LPC + HPC', hiKey:'hi_comp_min' },
  combustor:  { title:'Burner',                hiKey:null },
  turbine:    { title:'Turbine: HPT + LPT',    hiKey:'hi_turb_min' },
  nozzle:     { title:'Nozzle / Exhaust',      hiKey:null },
};

function trendText(hiKey){
  const i = currentCycle, win = 20;
  const i0 = Math.max(0, i-win);
  const v0 = data[i0][hiKey], v1 = data[i][hiKey];
  const span = Math.max(1, i-i0);
  const slope = (v1-v0)/span;
  let nextBoundary = null;
  for(const b of bandEdges){ if(v1>b+0.001){ nextBoundary=b; break; } }
  if(!(slope < -0.0003)){
    return `Stable over the last ${span} cycles (&Delta;${(slope*10).toFixed(3)}/10 cyc). No near-term threshold crossing projected.`;
  }
  if(nextBoundary===null){
    return `Declining ${(Math.abs(slope)*10).toFixed(3)}/10 cyc — already in the lowest health band.`;
  }
  const cyclesTo = Math.max(0, Math.round((v1-nextBoundary)/(-slope)));
  return `Declining ${(Math.abs(slope)*10).toFixed(3)}/10 cyc — est. <b>${cyclesTo}</b> cycles to next threshold (${bandLabel(nextBoundary-0.001)}).`;
}

function refreshComponentCard(){
  if(!selectedComponent || !data.length) return;
  const d = data[currentCycle];
  const info = compInfo[selectedComponent];
  safeGet('cc-title').textContent = info.title;
  const dot = safeGet('cc-dot');
  const statusEl = safeGet('cc-status');
  const m = (label, val) => `<div><div class="m-label">${label}</div><div class="m-val">${val}</div></div>`;
  const kel = v => isFinite(v) ? Math.round(v).toLocaleString() + ' K' : '—';
  const kpa = v => isFinite(v) ? Math.round(v).toLocaleString() + ' kPa' : '—';
  const win = data.slice(Math.max(0,currentCycle-40), currentCycle+1);
  let metricsHTML = '', trend = '', sparkArr = [], sparkColor = '#5fb3d9', base = null;
  if (info.hiKey) {
    const hi = d[info.hiKey];
    dot.style.background = bandColorRaw(hi);
    statusEl.textContent = bandLabel(hi) + ' — HI ' + fmtN(hi, 2);
    sparkColor = bandColorRaw(hi);
    sparkArr = win.map(x => x[info.hiKey]);
    trend = trendText(info.hiKey);
    base = 1.0;
  } else {
    dot.style.background = 'var(--ink-faint)';
    statusEl.textContent = 'Not health-tracked: N-CMAPSS has no health parameter here';
  }
  if (selectedComponent === 'inlet') {
    metricsHTML = m('Fan HI', fmtN(d.hi.fan,2)) + m('Fan PR', fmtN(d.fan_pr,3)) + m('Nf', Math.round(d.nf) + ' rpm') + m('Nf corr.', Math.round(d.nf_corr) + ' rpm');
  } else if (selectedComponent === 'compressor') {
    metricsHTML = m('LPC HI', fmtN(d.hi.lpc,2)) + m('HPC HI', fmtN(d.hi.hpc,2)) + m('HPC &eta; (proxy)', fmtN(d.eta_c,3)) + m('T30', kel(d.t2));
  } else if (selectedComponent === 'turbine') {
    metricsHTML = m('HPT HI', fmtN(d.hi.hpt,2)) + m('LPT HI', fmtN(d.hi.lpt,2)) + m('T48 (EGT)', kel(d.t48)) + m('T50', kel(d.t4));
  } else if (selectedComponent === 'combustor') {
    metricsHTML = m('Fuel flow', fmtN(d.wf,3) + ' kg/s') + m('&Delta;P/P (proxy)', fmtN(d.dp_comb*100,1) + '%') + m('T40 (est.)', kel(d.t3)) + m('P40', kpa(d.p3));
    trend = 'Burner deterioration is not a separate health parameter in N-CMAPSS; fuel-flow and temperature deviations appear in the HPT/LPT estimates.';
    sparkArr = win.map(x => x.res.Wf); sparkColor = '#e07b3f';
  } else {
    metricsHTML = m('T50', kel(d.t4)) + m('P50', kpa(d.p4));
    trend = 'Not health-tracked; LPT exit conditions shown.';
    sparkArr = win.map(x => x.t4);
  }
  safeGet('cc-metrics').innerHTML = metricsHTML;
  safeGet('cc-trend').innerHTML = trend + '<svg id="cc-spark" class="w-full h-6"></svg>';
  drawSpark(safeGet('cc-spark'), sparkArr, sparkColor, [200,24], base);
}
safeGet('cc-close').addEventListener('click', deselectComponent);

const hudAnchorsGLTF = {
  inlet:      new THREE.Vector3(320, 200, 0),
  compressor: new THREE.Vector3(75, 200, 0),
  combustor:  new THREE.Vector3(-202, 250, 0),
  turbine:    new THREE.Vector3(-475, 280, 0),
  nozzle:     new THREE.Vector3(-755, 300, 0)
};

const hudAnchorsSynthetic = {
  inlet:      new THREE.Vector3(-2.94, 0.9, 0),
  compressor: new THREE.Vector3(-1.62, 0.8, 0),
  combustor:  new THREE.Vector3(-0.245, 0.7, 0),
  turbine:    new THREE.Vector3(0.675, 0.65, 0),
  nozzle:     new THREE.Vector3(1.6, 0.6, 0)
};

function updateHUDLabels() {
  const gltfRoot = engineGroup.getObjectByName("gltfRoot");
  const anchors = gltfRoot ? hudAnchorsGLTF : hudAnchorsSynthetic;
  if (gltfRoot) { gltfRoot.updateMatrixWorld(true); } else { engineGroup.updateMatrixWorld(true); }
  const canvas = safeGet('engineCanvas');
  const width = canvas.clientWidth, height = canvas.clientHeight;
  const tempV = new THREE.Vector3();

  for (const key in anchors) {
    const el = document.getElementById(`hud-label-${key}`);
    if (!el) continue;

    tempV.copy(anchors[key]);
    let worldPos;
    if (gltfRoot) {
      worldPos = tempV.clone().applyMatrix4(gltfRoot.matrixWorld);
    } else {
      worldPos = tempV.clone().applyMatrix4(engineGroup.matrixWorld);
    }

    worldPos.project(camera);

    if (worldPos.z > 1) {
      el.style.display = 'none';
      continue;
    }

    el.style.display = 'flex';
    const x = (worldPos.x * 0.5 + 0.5) * width;
    const y = (-(worldPos.y * 0.5) + 0.5) * height;

    el.style.left = `${x}px`;
    el.style.top = `${y}px`;
  }
}

let lastFrame = performance.now();
let playbackAccum = 0;          // accumulated ms since last cycle advance
const CYCLE_INTERVAL_MS = 90;   // ms per cycle step (matches original 90 ms interval)
function animate(t){
  requestAnimationFrame(animate);
  const dt = (t - lastFrame) / 1000; lastFrame = t;

  if (Math.abs(currentPanOffset - targetPanOffset) > 0.5) {
    currentPanOffset += (targetPanOffset - currentPanOffset) * dt * 5.0;
    updateCamera();
  }

  // ---- Advance playback inside RAF so data updates never happen mid-draw ----
  if (playing) {
    playbackAccum += dt * 1000; // accumulate elapsed ms
    if (playbackAccum >= CYCLE_INTERVAL_MS) {
      playbackAccum -= CYCLE_INTERVAL_MS; // keep remainder, don't reset to 0
      let next = currentCycle + 1;
      if (next >= N) next = 0;
      setCycle(next);
    }
  }

  if(spoolGroup) spoolGroup.rotation.x += dt*3.4;
  [inletGroup, compressorGroup, turbineGroup].forEach(g=>{
    if(!g) return;
    g.children.forEach(m=>{ if(m.userData.isBlade) m.rotation.x += dt*3.4; });
  });
  if(nozzleGroup){
    nozzleGroup.children.forEach(m=>{
      if(m.userData.isPlume){ const s = 1 + 0.06*Math.sin(t*0.006); m.scale.set(s,1,s); }
    });
  }
  if(autoRotate && !isDragging){ rotY += dt*0.12; updateCamera(); }
  updateHUDLabels();
  renderer.render(scene, camera);
}

/* ============ Brayton Cycle Comparison Module ============ */
const CP_COLD = 1005;   // J/(kg·K) — cold section cp
const CP_HOT  = 1148;   // J/(kg·K) — hot  section cp
const GAMMA_C = 1.4;    // cold section gamma
const GAMMA_H = 1.33;   // hot  section gamma

function computeIdealBrayton(d) {
  // Station 1: Ambient stagnation (after ram compression)
  const T1 = d.ttamb;
  const P1 = d.ptamb;
  const s1 = 0; // reference entropy = 0 at station 1

  // Station 2s: Isentropic compression (ideal compressor)
  const T2s = T1 * Math.pow(d.pi_c, (GAMMA_C - 1) / GAMMA_C);
  const P2s = P1 * d.pi_c;
  const s2s = s1; // isentropic → no entropy change

  // Station 3: Constant-pressure heat addition to same T3 as actual
  const T3 = d.t3;
  const P3_ideal = P2s; // no combustor pressure drop in ideal cycle
  const s3_ideal = s2s + CP_HOT * Math.log(T3 / T2s); // ΔS = cp·ln(T3/T2)

  // Station 4s: Isentropic expansion (ideal turbine)
  // For ideal cycle, full expansion back to P1
  const pi_t_ideal = P3_ideal / P1;
  const T4s_ideal = T3 * Math.pow(pi_t_ideal, -(GAMMA_H - 1) / GAMMA_H);
  const P4s_ideal = P1;
  const s4s_ideal = s3_ideal; // isentropic → no entropy change

  // Ideal thermal efficiency: η_th = 1 - T1/T2s = 1 - 1/r_p^((γ-1)/γ)
  const eta_th_ideal = 1 - T1 / T2s;

  // Ideal specific work
  const w_comp_ideal = CP_COLD * (T2s - T1);
  const w_turb_ideal = CP_HOT * (T3 - T4s_ideal);
  const w_net_ideal = w_turb_ideal - w_comp_ideal;
  const q_in_ideal = CP_HOT * (T3 - T2s);

  return {
    stations: [
      { label: '1 (Inlet)',      T: T1,         P: P1,           s: s1 },
      { label: '2s (Comp exit)', T: T2s,        P: P2s,          s: s2s },
      { label: '3 (Comb exit)',  T: T3,          P: P3_ideal,     s: s3_ideal },
      { label: '4s (Turb exit)', T: T4s_ideal,  P: P4s_ideal,    s: s4s_ideal },
    ],
    eta_th: eta_th_ideal,
    w_comp: w_comp_ideal,
    w_turb: w_turb_ideal,
    w_net: w_net_ideal,
    q_in: q_in_ideal,
    bwr: w_comp_ideal / w_turb_ideal   // back-work ratio
  };
}

function computeActualBrayton(d) {
  const T1 = d.ttamb;
  const P1 = d.ptamb;
  const s1 = 0;

  // Station 2: Actual compressor (irreversible)
  const T2 = d.t2;
  const P2 = d.p2;
  const s2 = s1 + CP_COLD * Math.log(T2 / T1) - 287 * Math.log(P2 / P1); // Δs = cp·ln(T2/T1) - R·ln(P2/P1)

  // Station 3: Actual combustor exit
  const T3 = d.t3;
  const P3 = d.p3;
  const s3 = s2 + CP_HOT * Math.log(T3 / T2) - 287 * Math.log(P3 / P2);

  // Station 4: Actual turbine exit
  const T4 = d.t4;
  const P4 = d.p4;
  const s4 = s3 + CP_HOT * Math.log(T4 / T3) - 287 * Math.log(P4 / P3);

  const w_comp = CP_COLD * (T2 - T1);
  const w_turb = CP_HOT * (T3 - T4);
  const w_net = w_turb - w_comp;
  const q_in = CP_HOT * (T3 - T2);
  const eta_th = w_net / q_in;

  return {
    stations: [
      { label: '1 (Inlet)',      T: T1, P: P1, s: s1 },
      { label: '2 (Comp exit)',  T: T2, P: P2, s: s2 },
      { label: '3 (Comb exit)',  T: T3, P: P3, s: s3 },
      { label: '4 (Turb exit)',  T: T4, P: P4, s: s4 },
    ],
    eta_th,
    w_comp,
    w_turb,
    w_net,
    q_in,
    bwr: w_comp / w_turb
  };
}

function drawBraytonTs(ideal, actual) {
  const cvs = safeGet('braytonTsCanvas');
  if (!cvs) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = cvs.parentElement.getBoundingClientRect();
  cvs.width = rect.width * dpr;
  cvs.height = rect.height * dpr;
  const bctx = cvs.getContext('2d');
  bctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const w = rect.width, h = rect.height;

  bctx.clearRect(0, 0, w, h);

  const padL = 56, padR = 24, padT = 22, padB = 32;
  const iw = w - padL - padR, ih = h - padT - padB;
  const R_AIR = 287;
  const NPTS = 40; // interpolation points per process segment

  // --- Generate smooth curve points for a process between two stations ---
  function processCurve(T0, s0, T1, s1, npts) {
    const pts = [];
    const lnTRatio = Math.log(T1 / T0);
    const C = Math.abs(lnTRatio) > 1e-6 ? (s1 - s0) / lnTRatio : 0;
    
    for (let k = 0; k <= npts; k++) {
      const frac = k / npts;
      const T = T0 + (T1 - T0) * frac;
      let s;
      if (Math.abs(lnTRatio) > 1e-6) {
        s = s0 + C * Math.log(T / T0);
      } else {
        s = s0 + (s1 - s0) * frac;
      }
      pts.push({ T, s });
    }
    return pts;
  }

  // --- Build smooth curve arrays ---
  const iS = ideal.stations;
  const idealCurve = [
    ...processCurve(iS[0].T, iS[0].s, iS[1].T, iS[1].s, NPTS),
    ...processCurve(iS[1].T, iS[1].s, iS[2].T, iS[2].s, NPTS).slice(1),
    ...processCurve(iS[2].T, iS[2].s, iS[3].T, iS[3].s, NPTS).slice(1),
    ...processCurve(iS[3].T, iS[3].s, iS[0].T, iS[0].s, NPTS).slice(1),
  ];

  const aS = actual.stations;
  const actualCurve = [
    ...processCurve(aS[0].T, aS[0].s, aS[1].T, aS[1].s, NPTS),
    ...processCurve(aS[1].T, aS[1].s, aS[2].T, aS[2].s, NPTS).slice(1),
    ...processCurve(aS[2].T, aS[2].s, aS[3].T, aS[3].s, NPTS).slice(1),
    ...processCurve(aS[3].T, aS[3].s, aS[0].T, aS[0].s, NPTS).slice(1),
  ];

  // --- Axis scaling ---
  const allPts = [...idealCurve, ...actualCurve];
  const allT = allPts.map(p => p.T);
  const allS = allPts.map(p => p.s);
  const Tmin = Math.min(...allT) - 40, Tmax = Math.max(...allT) + 60;
  const Smin = Math.min(...allS) - 30, Smax = Math.max(...allS) + 50;

  const xMap = s => padL + (s - Smin) / (Smax - Smin) * iw;
  const yMap = T => padT + (1 - (T - Tmin) / (Tmax - Tmin)) * ih;

  // --- Background grid ---
  bctx.strokeStyle = 'rgba(0,0,0,0.1)';
  bctx.lineWidth = 0.7;
  for (let g = 0; g <= 6; g++) {
    const y = padT + g / 6 * ih;
    const Tval = Tmax - g / 6 * (Tmax - Tmin);
    bctx.beginPath(); bctx.moveTo(padL, y); bctx.lineTo(w - padR, y); bctx.stroke();
    bctx.fillStyle = '#8d9c91';
    bctx.font = '9px IBM Plex Mono';
    bctx.textAlign = 'right';
    bctx.fillText(Math.round(Tval), padL - 6, y + 3);
  }
  for (let g = 0; g <= 5; g++) {
    const x = padL + g / 5 * iw;
    const Sval = Smin + g / 5 * (Smax - Smin);
    bctx.beginPath(); bctx.moveTo(x, padT); bctx.lineTo(x, padT + ih); bctx.stroke();
    bctx.fillStyle = '#8d9c91';
    bctx.textAlign = 'center';
    bctx.fillText(Math.round(Sval), x, padT + ih + 14);
  }
  // Axis border lines
  bctx.strokeStyle = 'rgba(0,0,0,0.3)';
  bctx.lineWidth = 1.2;
  bctx.beginPath();
  bctx.moveTo(padL, padT); bctx.lineTo(padL, padT + ih); bctx.lineTo(w - padR, padT + ih);
  bctx.stroke();

  // --- Background isobar reference curves ---
  const isobarPressures = [iS[0].P, iS[1].P]; // ambient P and compressor-exit P
  bctx.save();
  bctx.setLineDash([3, 4]);
  bctx.lineWidth = 0.8;
  isobarPressures.forEach((Pref, idx) => {
    bctx.strokeStyle = idx === 0 ? 'rgba(95,179,217,0.12)' : 'rgba(79,209,181,0.12)';
    bctx.beginPath();
    let started = false;
    for (let k = 0; k <= 60; k++) {
      const sFrac = Smin + (k / 60) * (Smax - Smin);
      // At constant P: T = Tref * exp((s - sRef)/cp)
      // Use station 1 as reference for the low isobar, station 2s for the high
      const Tref = idx === 0 ? iS[0].T : iS[1].T;
      const sRef = idx === 0 ? iS[0].s : iS[1].s;
      const cp = idx === 0 ? CP_COLD : ((sFrac > (iS[1].s + iS[2].s) / 2) ? CP_HOT : CP_COLD);
      const Tcurve = Tref * Math.exp((sFrac - sRef) / cp);
      if (Tcurve < Tmin || Tcurve > Tmax) { started = false; continue; }
      const px = xMap(sFrac), py = yMap(Tcurve);
      if (!started) { bctx.moveTo(px, py); started = true; } else { bctx.lineTo(px, py); }
    }
    bctx.stroke();
  });
  bctx.setLineDash([]);
  bctx.restore();

  // --- Isobar labels ---
  bctx.save();
  bctx.font = '500 7.5px IBM Plex Mono';
  bctx.globalAlpha = 0.3;
  bctx.fillStyle = '#5fb3d9';
  const pLowLabel = (iS[0].P / 1000).toFixed(0) + ' kPa';
  const pHighLabel = (iS[1].P / 1000).toFixed(0) + ' kPa';
  // Position labels near the right edge of the isobars
  const labelSPos = Smax - (Smax - Smin) * 0.15;
  const TlowIso = iS[0].T * Math.exp((labelSPos - iS[0].s) / CP_COLD);
  const ThighIso = iS[1].T * Math.exp((labelSPos - iS[1].s) / CP_HOT);
  if (TlowIso >= Tmin && TlowIso <= Tmax) {
    bctx.textAlign = 'right';
    bctx.fillText('P = ' + pLowLabel, xMap(labelSPos), yMap(TlowIso) - 4);
  }
  bctx.fillStyle = '#4fd1b5';
  if (ThighIso >= Tmin && ThighIso <= Tmax) {
    bctx.textAlign = 'right';
    bctx.fillText('P = ' + pHighLabel, xMap(labelSPos), yMap(ThighIso) - 4);
  }
  bctx.restore();

  // ============ LAYER 1: IDEAL CYCLE (background, faded filled shape) ============
  bctx.save();
  // Filled area
  bctx.globalAlpha = 0.07;
  bctx.fillStyle = '#5fb3d9';
  bctx.beginPath();
  idealCurve.forEach((p, i) => {
    const px = xMap(p.s), py = yMap(p.T);
    if (i === 0) bctx.moveTo(px, py); else bctx.lineTo(px, py);
  });
  bctx.closePath();
  bctx.fill();

  // Outline
  bctx.globalAlpha = 0.35;
  bctx.strokeStyle = '#5fb3d9';
  bctx.lineWidth = 1.8;
  bctx.setLineDash([]);
  bctx.beginPath();
  idealCurve.forEach((p, i) => {
    const px = xMap(p.s), py = yMap(p.T);
    if (i === 0) bctx.moveTo(px, py); else bctx.lineTo(px, py);
  });
  bctx.closePath();
  bctx.stroke();
  bctx.restore();

  // Ideal station markers (small, muted)
  bctx.save();
  bctx.globalAlpha = 0.45;
  iS.forEach((st, i) => {
    const px = xMap(st.s), py = yMap(st.T);
    bctx.fillStyle = '#5fb3d9';
    bctx.beginPath(); bctx.arc(px, py, 3.5, 0, Math.PI * 2); bctx.fill();
    bctx.fillStyle = '#ffffff';
    bctx.beginPath(); bctx.arc(px, py, 1.5, 0, Math.PI * 2); bctx.fill();
    // Label
    bctx.fillStyle = '#5fb3d9';
    bctx.font = '600 8px IBM Plex Sans';
    const lbl = (i + 1) + (i === 1 ? 's' : i === 3 ? 's' : '');
    bctx.textAlign = (i === 0 || i === 3) ? 'right' : 'left';
    const offX = (i === 0 || i === 3) ? -7 : 7;
    const offY = (i <= 1) ? -6 : 10;
    bctx.fillText(lbl, px + offX, py + offY);
  });
  bctx.restore();

  // "IDEAL" label in the center of the ideal cycle
  bctx.save();
  bctx.globalAlpha = 0.18;
  bctx.fillStyle = '#5fb3d9';
  bctx.font = '700 11px IBM Plex Sans Condensed';
  bctx.textAlign = 'center';
  const idealCenterS = (iS[0].s + iS[2].s) / 2;
  const idealCenterT = (iS[0].T + iS[2].T) / 2;
  bctx.fillText('IDEAL', xMap(idealCenterS), yMap(idealCenterT) + 4);
  bctx.restore();

  // ============ LAYER 2: LOSS SHADING between ideal and actual ============
  // Compression loss shading (between 1→2s and 1→2)
  bctx.save();
  bctx.globalAlpha = 0.10;
  bctx.fillStyle = '#ea580c';
  const compIdealPts = processCurve(iS[0].T, iS[0].s, iS[1].T, iS[1].s, NPTS);
  const compActualPts = processCurve(aS[0].T, aS[0].s, aS[1].T, aS[1].s, NPTS);
  bctx.beginPath();
  compIdealPts.forEach((p, i) => {
    const px = xMap(p.s), py = yMap(p.T);
    if (i === 0) bctx.moveTo(px, py); else bctx.lineTo(px, py);
  });
  for (let i = compActualPts.length - 1; i >= 0; i--) {
    bctx.lineTo(xMap(compActualPts[i].s), yMap(compActualPts[i].T));
  }
  bctx.closePath();
  bctx.fill();

  // Expansion loss shading (between 3→4s and 3→4)
  const expIdealPts = processCurve(iS[2].T, iS[2].s, iS[3].T, iS[3].s, NPTS);
  const expActualPts = processCurve(aS[2].T, aS[2].s, aS[3].T, aS[3].s, NPTS);
  bctx.beginPath();
  expIdealPts.forEach((p, i) => {
    const px = xMap(p.s), py = yMap(p.T);
    if (i === 0) bctx.moveTo(px, py); else bctx.lineTo(px, py);
  });
  for (let i = expActualPts.length - 1; i >= 0; i--) {
    bctx.lineTo(xMap(expActualPts[i].s), yMap(expActualPts[i].T));
  }
  bctx.closePath();
  bctx.fill();
  bctx.restore();

  // ============ LAYER 3: ACTUAL CYCLE (foreground, prominent) ============
  // Filled area (subtle)
  bctx.save();
  bctx.globalAlpha = 0.09;
  bctx.fillStyle = '#4fd1b5';
  bctx.beginPath();
  actualCurve.forEach((p, i) => {
    const px = xMap(p.s), py = yMap(p.T);
    if (i === 0) bctx.moveTo(px, py); else bctx.lineTo(px, py);
  });
  bctx.closePath();
  bctx.fill();
  bctx.restore();

  // Glow effect
  bctx.save();
  bctx.strokeStyle = '#4fd1b5';
  bctx.lineWidth = 6;
  bctx.globalAlpha = 0.08;
  bctx.lineJoin = 'round';
  bctx.beginPath();
  actualCurve.forEach((p, i) => {
    const px = xMap(p.s), py = yMap(p.T);
    if (i === 0) bctx.moveTo(px, py); else bctx.lineTo(px, py);
  });
  bctx.closePath();
  bctx.stroke();
  bctx.restore();

  // Main stroke
  bctx.save();
  bctx.strokeStyle = '#4fd1b5';
  bctx.lineWidth = 2.8;
  bctx.lineJoin = 'round';
  bctx.lineCap = 'round';
  bctx.shadowColor = '#4fd1b5';
  bctx.shadowBlur = 6;
  bctx.beginPath();
  actualCurve.forEach((p, i) => {
    const px = xMap(p.s), py = yMap(p.T);
    if (i === 0) bctx.moveTo(px, py); else bctx.lineTo(px, py);
  });
  bctx.closePath();
  bctx.stroke();
  bctx.shadowBlur = 0;
  bctx.restore();

  // Actual station markers (prominent, with pulse ring)
  aS.forEach((st, i) => {
    const px = xMap(st.s), py = yMap(st.T);
    // Outer pulse ring
    bctx.save();
    bctx.globalAlpha = 0.18;
    bctx.strokeStyle = '#4fd1b5';
    bctx.lineWidth = 1.5;
    bctx.beginPath(); bctx.arc(px, py, 9, 0, Math.PI * 2); bctx.stroke();
    bctx.restore();
    // Solid dot
    bctx.fillStyle = '#4fd1b5';
    bctx.beginPath(); bctx.arc(px, py, 5, 0, Math.PI * 2); bctx.fill();
    bctx.fillStyle = 'rgba(255,255,255,0.2)';
    bctx.beginPath(); bctx.arc(px, py, 2.2, 0, Math.PI * 2); bctx.fill();
    // Label
    bctx.fillStyle = '#d7dee6';
    bctx.font = '700 10px IBM Plex Sans';
    const lbl = String(i + 1);
    bctx.textAlign = (i === 0 || i === 3) ? 'right' : 'left';
    const offX = (i === 0 || i === 3) ? -12 : 12;
    const offY = (i <= 1) ? -8 : 12;
    bctx.fillText(lbl, px + offX, py + offY);
    // Temperature readout next to station
    bctx.font = '500 8.5px IBM Plex Mono';
    bctx.fillStyle = '#7c8896';
    bctx.fillText(Math.round(st.T) + ' K', px + offX, py + offY + 11);
  });

  // ============ LAYER 4: PROCESS ANNOTATIONS ============
  bctx.save();
  bctx.font = '700 8.5px IBM Plex Mono';
  bctx.globalAlpha = 0.50;

  // Compression (1→2) — left side, along the rising curve
  const compMidIdx = Math.floor(compActualPts.length / 2);
  const compMid = compActualPts[compMidIdx];
  bctx.fillStyle = '#5fb3d9';
  bctx.textAlign = 'right';
  bctx.fillText('COMPRESSION', xMap(compMid.s) - 10, yMap(compMid.T));

  // Heat addition (2→3) — top, along the rightward curve
  const heatPts = processCurve(aS[1].T, aS[1].s, aS[2].T, aS[2].s, NPTS);
  const heatMid = heatPts[Math.floor(heatPts.length / 2)];
  bctx.fillStyle = '#f59e0b';
  bctx.textAlign = 'center';
  bctx.fillText('HEAT ADDITION', xMap(heatMid.s), yMap(heatMid.T) - 10);

  // Expansion (3→4) — right side, along the falling curve
  const expMidIdx = Math.floor(expActualPts.length / 2);
  const expMid = expActualPts[expMidIdx];
  bctx.fillStyle = '#4fd1b5';
  bctx.textAlign = 'left';
  bctx.fillText('EXPANSION', xMap(expMid.s) + 10, yMap(expMid.T));

  // Heat rejection (4→1) — bottom
  const rejPts = processCurve(aS[3].T, aS[3].s, aS[0].T, aS[0].s, NPTS);
  const rejMid = rejPts[Math.floor(rejPts.length / 2)];
  bctx.fillStyle = '#dc2626';
  bctx.textAlign = 'center';
  bctx.fillText('HEAT REJECTION', xMap(rejMid.s), yMap(rejMid.T) + 14);
  bctx.restore();

  // ============ LAYER 5: AXIS LABELS ============
  bctx.fillStyle = '#4a5561';
  bctx.font = '600 8.5px IBM Plex Mono';
  bctx.textAlign = 'center';
  bctx.fillText('K', padL - 6, padT - 6);
  bctx.fillText('J/(kg·K)', w - padR, padT + ih + 24);
}

function updateBraytonEffCards(ideal, actual) {
  const grid = safeGet('effCompareGrid');
  if (!grid) return;

  const cards = [
    { label: 'Thermal Efficiency',      idealVal: (ideal.eta_th * 100).toFixed(1) + '%',  actualVal: (actual.eta_th * 100).toFixed(1) + '%',  idealPct: ideal.eta_th, actualPct: actual.eta_th, delta: ((ideal.eta_th - actual.eta_th) * 100).toFixed(1) + ' pp' },
    { label: 'Net Specific Work',        idealVal: (ideal.w_net / 1000).toFixed(1) + ' kJ/kg', actualVal: (actual.w_net / 1000).toFixed(1) + ' kJ/kg', idealPct: 1, actualPct: actual.w_net / ideal.w_net, delta: ((1 - actual.w_net / ideal.w_net) * 100).toFixed(1) + '%' },
    { label: 'HPC η_is (proxy)',          idealVal: '100.0%', actualVal: (data[currentCycle].eta_c * 100).toFixed(1) + '%', idealPct: 1, actualPct: data[currentCycle].eta_c, delta: ((1 - data[currentCycle].eta_c) * 100).toFixed(1) + ' pp' },
    { label: 'Turbine η_is (est.)',             idealVal: '100.0%', actualVal: (data[currentCycle].eta_t * 100).toFixed(1) + '%', idealPct: 1, actualPct: data[currentCycle].eta_t, delta: ((1 - data[currentCycle].eta_t) * 100).toFixed(1) + ' pp' },
    { label: 'Back-Work Ratio',          idealVal: (ideal.bwr * 100).toFixed(1) + '%', actualVal: (actual.bwr * 100).toFixed(1) + '%', idealPct: ideal.bwr, actualPct: actual.bwr, delta: ((actual.bwr - ideal.bwr) * 100).toFixed(1) + ' pp' },
    { label: 'Burner ΔP/P (proxy)',      idealVal: '0.0%', actualVal: (data[currentCycle].dp_comb * 100).toFixed(1) + '%', idealPct: 0, actualPct: data[currentCycle].dp_comb * 3, delta: (data[currentCycle].dp_comb * 100).toFixed(1) + '%' },
  ];

  grid.innerHTML = cards.map(c => `
    <div class="eff-card">
      <div class="eff-label">${c.label}</div>
      <div class="eff-vals">
        <span class="eff-ideal">${c.idealVal}</span>
        <span class="eff-actual">${c.actualVal}</span>
        <span class="eff-delta">Δ ${c.delta}</span>
      </div>
      <div class="eff-bar-track">
        <div class="eff-bar-ideal" style="width:${Math.min(100, c.idealPct * 100).toFixed(0)}%"></div>
        <div class="eff-bar-actual" style="width:${Math.min(100, c.actualPct * 100).toFixed(0)}%"></div>
      </div>
    </div>
  `).join('');
}

function updateBraytonTable() {
  // Layer 2: healthy-engine reference ("ideal" at the same flight condition) vs measured, per sensor
  const tbody = safeGet('braytonTableBody');
  if (!tbody || !data.length) return;
  const d = data[currentCycle];
  const rows = ['T24','T30','T48','T50','P24','Ps30','P40','P50','Nf','Nc','Wf'];
  const unit = s => s.startsWith('T') ? 'K' : (s.startsWith('P') ? 'kPa' : (s === 'Wf' ? 'kg/s' : 'rpm'));
  const dp = s => s === 'Wf' ? 3 : (s.startsWith('P') ? 1 : 0);
  tbody.innerHTML = rows.map(s => {
    const r = d.res[s];
    return `<tr>
      <td class="station-col" title="${(backend.payload && backend.payload.sensor_info[s]) || ''}">${s}</td>
      <td class="ideal-col">${fmtN(d.ideal[s], dp(s))} ${unit(s)}</td>
      <td class="actual-col">${fmtN(d.x[s], dp(s))} ${unit(s)}</td>
      <td class="${Math.abs(r) > 1 ? 'loss-col' : ''}">${pct(r)}</td>
    </tr>`;
  }).join('');
}

function updateBraytonSummary(ideal, actual) {
  const row = safeGet('braytonSummaryRow');
  if (!row) return;

  const items = [
    { label: 'η_th Ideal',     val: (ideal.eta_th * 100).toFixed(1) + '%',       cls: 'ideal' },
    { label: 'η_th Actual',    val: (actual.eta_th * 100).toFixed(1) + '%',      cls: 'actual' },
    { label: 'W_net Ideal',    val: (ideal.w_net / 1000).toFixed(1) + ' kJ/kg',  cls: 'ideal' },
    { label: 'W_net Actual',   val: (actual.w_net / 1000).toFixed(1) + ' kJ/kg', cls: 'actual' },
    { label: 'Q_in',           val: (ideal.q_in / 1000).toFixed(1) + ' kJ/kg',   cls: 'ideal' },
    { label: 'Efficiency Loss', val: ((ideal.eta_th - actual.eta_th) * 100).toFixed(1) + ' pp', cls: '' },
  ];

  row.innerHTML = items.map(it => `
    <div class="brayton-summary-item">
      <div class="bs-label">${it.label}</div>
      <div class="bs-val ${it.cls}" ${!it.cls ? 'style="color:var(--orange)"' : ''}>${it.val}</div>
    </div>
  `).join('');
}

function updateBraytonModule(d) {
  updateBraytonTable();
  const ok = [d.ttamb, d.ptamb, d.t2, d.p2, d.t3, d.p3, d.t4, d.p4].every(v => isFinite(v) && v > 0);
  if (!ok) {                       // needs the (virtual) burner-exit temperature T40
    const cvs = safeGet('braytonTsCanvas');
    const c2 = cvs.getContext && cvs.getContext('2d');
    if (c2) c2.clearRect(0, 0, cvs.width, cvs.height);
    return;
  }
  const ideal = computeIdealBrayton(d);
  const actual = computeActualBrayton(d);
  drawBraytonTs(ideal, actual);
  updateBraytonEffCards(ideal, actual);
  updateBraytonSummary(ideal, actual);
}

/* ============ Render frame ============ */
function setCheck(id, ok, okText, badText) {
  const el = safeGet(id);
  el.textContent = ok ? okText : badText;
  el.style.color = ok ? 'var(--teal)' : 'var(--orange)';
  el.style.borderColor = ok ? 'rgba(79,209,181,0.3)' : 'rgba(224,123,63,0.3)';
  el.style.background = ok ? 'rgba(79,209,181,0.05)' : 'rgba(224,123,63,0.05)';
}

function renderComponents(i) {
  const box = document.getElementById('component-cards');
  if (!box) return;
  const d = data[i];
  const s = Math.max(0, i - 40);
  box.innerHTML = COMPONENTS.map(cc => {
    const hi = d.hi[cc], truth = d.truth[cc];
    return `<div class="flex items-center gap-3 py-2 border-b border-outline-variant/40 last:border-0 cursor-pointer hover:bg-surface-container-low/60 rounded" onclick="selectComponent('${COMP_SECTION[cc]}')">
      <div class="relative w-16 h-11 shrink-0">
        <svg id="gauge-${cc}" class="w-full h-full" viewBox="0 0 130 76"></svg>
        <div class="absolute inset-x-0 bottom-0 text-center font-data-display text-[12px] font-bold" style="color:${bandColorRaw(hi)}">${fmtN(hi, 2)}</div>
      </div>
      <div class="flex-1 min-w-0">
        <div class="flex justify-between text-[11px] font-label-sm"><span class="font-bold text-on-surface">${COMP_LABEL[cc]}</span><span style="color:${bandColorRaw(hi)}">${bandLabel(hi)}</span></div>
        <div class="text-[10px] font-label-sm text-on-surface-variant truncate">ML ${fmtN(hi,2)} ±${fmtN(d.spread[cc],2)} · KF ${fmtN(d.kf[cc],2)} ±${fmtN(d.kfs[cc],2)}${isFinite(truth) ? ' · true ' + fmtN(truth,2) : ''}</div>
        <svg id="spark-${cc}" class="w-full h-4"></svg>
      </div>
    </div>`;
  }).join('');
  COMPONENTS.forEach(cc => {
    drawGauge(document.getElementById(`gauge-${cc}`), d.hi[cc], isFinite(d.spread[cc]) ? d.spread[cc] : 0);
    drawSpark(document.getElementById(`spark-${cc}`), data.slice(s, i + 1).map(x => x.hi[cc]), bandColorRaw(d.hi[cc]), [200, 16], 1.0);
  });
}

function render(i){
  if (!data.length) return;
  const d = data[i];

  const cycleInput = safeGet('cycleInput');
  if (cycleInput && document.activeElement !== cycleInput) {
    cycleInput.value = i + 1;
    cycleInput.max = N;
  }
  const maxSpan = safeGet('maxCycleSpan');
  if (maxSpan) maxSpan.textContent = ` / ${N}`;

  // 01 flight state (cycle means)
  safeGet('r-alt').innerHTML = (isFinite(d.alt) ? Math.round(d.alt).toLocaleString() : '—') + '<span class="unit">m</span>';
  safeGet('r-mach').textContent = fmtN(d.mach, 2);
  safeGet('r-tra').innerHTML = fmtN(d.tra, 1) + '<span class="unit">%</span>';
  safeGet('r-t2').innerHTML = fmtN(d.ttamb, 0) + '<span class="unit">K</span>';
  safeGet('r-nf').innerHTML = (isFinite(d.nf) ? Math.round(d.nf).toLocaleString() : '—') + '<span class="unit">rpm</span>';
  safeGet('r-nc').innerHTML = (isFinite(d.nc) ? Math.round(d.nc).toLocaleString() : '—') + '<span class="unit">rpm</span>';
  safeGet('r-wf').innerHTML = fmtN(d.wf, 3) + '<span class="unit">kg/s</span>';
  safeGet('r-fc').innerHTML = (d.fc ?? '—') + `<span class="unit">· ${fmtN(d.duration, 0)} min</span>`;

  // 09 physics & interpretability
  safeGet('p-eta-hpc').textContent = fmtN(d.eta_c, 3);
  safeGet('p-eta-core').textContent = fmtN(d.eta_core, 3);
  safeGet('p-egt').textContent = pct(d.res.T48);
  safeGet('p-wf').textContent = pct(d.res.Wf);
  safeGet('p-eta-turb').textContent = isFinite(d.eta_t) ? fmtN(d.eta_t, 3) + ' (est.)' : '—';
  const rawOk = d.hiRaw.every(v => !isFinite(v) || (v > -0.05 && v < 1.05));
  const prev = i > 0 ? data[i - 1] : null;
  const mono = !prev || COMPONENTS.every(cc => !(d.hi[cc] > prev.hi[cc] + 0.05));
  const kfOk = Object.entries(d.theta).filter(([k]) => k.endsWith('eff_mod')).every(([, v]) => !(v > 0.005));
  setCheck('chk-eff', rawOk, 'PASSED', 'FAILED');
  setCheck('chk-mono', mono, 'PASSED', 'VIOLATION');
  setCheck('chk-kf', kfOk, 'PASSED', 'CHECK');

  // 02 component health
  renderComponents(i);

  // 05 overall health = weakest component
  const circumference = 263.8;
  const ring = safeGet('ohiRing');
  ring.style.strokeDashoffset = circumference * (1 - Math.max(0, Math.min(1, d.ohi)));
  ring.style.stroke = bandColorRaw(d.ohi);
  safeGet('ohiNum').innerHTML = `${fmtN(d.ohi, 2)}<div style="font-size:9.5px; font-weight:500; opacity:0.8; margin-top:-2px;">&plusmn;${fmtN(d.u_ohi, 2)}</div>`;
  const label = bandLabel(d.ohi);
  safeGet('statusDot').style.background = bandColorRaw(d.ohi);
  safeGet('statusText').textContent = label;
  safeGet('ohiDesc').textContent = `${bandDesc[label]} Weakest component: ${COMP_LABEL[d.ohiComp] || '—'}.`;
  const marker = safeGet('ohi-marker');
  marker.style.right = 'auto';
  marker.style.left = `calc(${Math.max(0, Math.min(1, d.ohi)) * 100}% - 4px)`;

  // 06 fuel-flow penalty vs healthy engine (efficiency loss shows up as extra fuel)
  const win = 40, s = Math.max(0, i - win);
  safeGet('thrustVal').textContent = pct(d.res.Wf);
  const deltaEl = safeGet('thrustDelta');
  deltaEl.textContent = `${fmtN(d.wf, 3)} kg/s measured vs ${fmtN(d.wfIdeal, 3)} kg/s healthy`;
  deltaEl.style.color = d.res.Wf > 1 ? 'var(--orange)' : 'var(--ink-dim)';
  drawSpark(safeGet('spark-thrust'), data.slice(s, i + 1).map(x => x.res.Wf), '#e07b3f', [400, 46], 0);

  // 08 RUL with 90% interval
  const width = d.q95 - d.q05;
  const sig = isFinite(width) ? width / 3.29 : NaN;
  const conf = isFinite(sig) ? Math.max(0, Math.min(100, Math.round(100 * (1 - sig / Math.max(1, d.rul))))) : null;
  safeGet('confNum').textContent = conf === null ? '—' : conf + '%';
  safeGet('confNum').style.color = conf === null ? 'var(--ink-dim)' : conf > 70 ? 'var(--teal)' : conf > 45 ? 'var(--amber)' : 'var(--orange)';
  safeGet('confBar').style.width = (conf || 0) + '%';
  safeGet('rulVal').textContent = fmtN(d.rul, 0);
  safeGet('sigmaVal').textContent = (isFinite(width) ? `90%: ${fmtN(d.q05, 0)}–${fmtN(d.q95, 0)} cyc` : '—') +
    (isFinite(d.truthRul) ? ` · true ${fmtN(d.truthRul, 0)}` : '');

  // 07 trends
  drawTrend(safeGet('trendChart'), i);
  const rulMax = Math.max(1, ...data.map(x => x.rul).filter(isFinite));
  [['trend-rul-max', 1], ['trend-rul-75', 0.75], ['trend-rul-50', 0.5], ['trend-rul-25', 0.25]].forEach(([id, f]) => {
    safeGet(id).textContent = Math.round(rulMax * f);
  });
  safeGet('trend-cyc-max').textContent = 'CYC ' + N;

  updateEngineVisuals(d);
  updateBraytonModule(d);
  if(selectedComponent) refreshComponentCard();
  if(typeof renderBackendPanels === 'function') renderBackendPanels(i);
}

/* ============ Anomaly Navigation Helpers ============ */
function getBandIndex(v) {
  return bandIdx(v);
}

function getAnomalyCycles() {
  const anomalies = [];
  if (!data || data.length === 0) return anomalies;
  for (let i = 1; i < N; i++) {
    if (getBandIndex(data[i].ohi) !== getBandIndex(data[i-1].ohi)) {
      anomalies.push(i);
    }
  }
  return anomalies;
}

function jumpToPrevAnomaly() {
  const anomalies = getAnomalyCycles();
  if (anomalies.length === 0) return;
  const prev = [...anomalies].reverse().find(c => c < currentCycle);
  if (prev !== undefined) {
    setCycle(prev);
  } else {
    setCycle(anomalies[anomalies.length - 1]);
  }
}

function jumpToNextAnomaly() {
  const anomalies = getAnomalyCycles();
  if (anomalies.length === 0) return;
  const next = anomalies.find(c => c > currentCycle);
  if (next !== undefined) {
    setCycle(next);
  } else {
    setCycle(anomalies[0]);
  }
}

/* ============ Tape scrubber ============ */
const canvas = safeGet('tapeCanvas');
const ctx = canvas.getContext('2d');
function sizeCanvas(){
  const dpr = window.devicePixelRatio||1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width*dpr; canvas.height = 52*dpr;
  ctx.setTransform(dpr,0,0,dpr,0,0);
  drawTape();
}
function drawTape(){
  if (!data.length) { ctx.clearRect(0, 0, canvas.width, canvas.height); return; }
  const rect = canvas.getBoundingClientRect();
  const w = rect.width, h = 52;
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = 'rgba(255,255,255,0.2)';
  ctx.fillRect(0,0,w,h);
  for(let i=0;i<N;i+=1){
    const x = i/(N-1)*w;
    const major = i%25===0;
    ctx.strokeStyle = major? '#3a4552' : '#1a222b';
    ctx.lineWidth = major?1.2:1;
    const th = major? 16: 8;
    ctx.beginPath(); ctx.moveTo(x,h-th); ctx.lineTo(x,h); ctx.stroke();
    if(major){
      ctx.fillStyle = '#4a5561';
      ctx.font = '9px IBM Plex Mono';
      ctx.textAlign='center';
      ctx.fillText(i, x, h-20);
    }
  }
  for(let i=0;i<N;i++){
    const x0 = i/(N-1)*w, x1=(i+1)/(N-1)*w;
    ctx.fillStyle = bandColorRaw(data[i].ohi);
    ctx.globalAlpha = 0.55;
    ctx.fillRect(x0, 2, Math.max(1,x1-x0), 4);
  }
  ctx.globalAlpha = 1;

  // Draw threshold crossing anomaly markers on tape track
  const anomalyCycles = getAnomalyCycles();
  for(let i=0; i<anomalyCycles.length; i++){
    const ac = anomalyCycles[i];
    const ax = ac/(N-1)*w;
    ctx.fillStyle = bandColorRaw(data[ac].ohi);
    ctx.beginPath();
    ctx.moveTo(ax-3.5, 0);
    ctx.lineTo(ax+3.5, 0);
    ctx.lineTo(ax, 5);
    ctx.closePath();
    ctx.fill();
    ctx.strokeStyle = bandColorRaw(data[ac].ohi);
    ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(ax, 5); ctx.lineTo(ax, h-16); ctx.stroke();
  }

  const px = currentCycle/(N-1)*w;
  ctx.strokeStyle = '#d7dee6';
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(px,0); ctx.lineTo(px,h); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(px-5,0); ctx.lineTo(px+5,0); ctx.lineTo(px,7); ctx.closePath();
  ctx.fillStyle = '#d7dee6'; ctx.fill();
}
window.addEventListener('resize', sizeCanvas);

function setCycle(i){
  if (!data.length) return;
  currentCycle = Math.max(0, Math.min(N-1, i));
  render(currentCycle);
  drawTape();
}

let dragging = false;
function cycleFromEvent(e){
  const rect = canvas.getBoundingClientRect();
  const x = (e.touches? e.touches[0].clientX : e.clientX) - rect.left;
  return Math.round(x/rect.width*(N-1));
}
canvas.addEventListener('mousedown', e=>{ dragging=true; setCycle(cycleFromEvent(e)); });
window.addEventListener('mousemove', e=>{ if(dragging) setCycle(cycleFromEvent(e)); });
window.addEventListener('mouseup', ()=> dragging=false);
canvas.addEventListener('touchstart', e=>{ dragging=true; setCycle(cycleFromEvent(e)); });
canvas.addEventListener('touchmove', e=>{ if(dragging) setCycle(cycleFromEvent(e)); });
window.addEventListener('touchend', ()=> dragging=false);

let playing = false;
safeGet('playBtn').addEventListener('click', ()=>{
  playing = !playing;
  playbackAccum = 0; // reset accumulator on play/pause
  safeGet('playBtn').innerHTML = playing? '&#10074;&#10074;' : '&#9654;';
});

// Editable cycle input handler
const cycleInput = safeGet('cycleInput');
if(cycleInput){
  cycleInput.addEventListener('change', (e) => {
    let val = parseInt(e.target.value, 10);
    if (!isNaN(val)) setCycle(val - 1);
  });
  cycleInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      let val = parseInt(e.target.value, 10);
      if (!isNaN(val)) setCycle(val - 1);
      e.target.blur();
    }
  });
}

// Global Keyboard Navigation
window.addEventListener('keydown', (e) => {
  const activeTag = document.activeElement ? document.activeElement.tagName.toLowerCase() : '';
  if (activeTag === 'input' || activeTag === 'select' || activeTag === 'textarea') return;

  if (e.key === 'ArrowLeft') {
    e.preventDefault();
    setCycle(currentCycle - (e.shiftKey ? 10 : 1));
  } else if (e.key === 'ArrowRight') {
    e.preventDefault();
    setCycle(currentCycle + (e.shiftKey ? 10 : 1));
  } else if (e.key === ' ' || e.code === 'Space') {
    e.preventDefault();
    safeGet('playBtn').click();
  }
});

// Dropdown handler
const engineSelect = document.querySelector('.engine-select');
function showEngine() {
  currentCycle = 0;
  const maxSpan = safeGet('maxCycleSpan');
  if (maxSpan) maxSpan.textContent = ` / ${N}`;
  sizeCanvas();
  deselectComponent();
  setCycle(0);
}
if (engineSelect) {
  engineSelect.addEventListener('change', async (e) => {
    try {
      await loadEngine(e.target.value);
      showEngine();
    } catch (err) {
      alert('Could not load engine: ' + err.message);
    }
  });
}

// Anomaly navigation buttons
const prevBtn = document.getElementById('prevAnomalyBtn');
if (prevBtn) prevBtn.addEventListener('click', jumpToPrevAnomaly);
const nextBtn = document.getElementById('nextAnomalyBtn');
if (nextBtn) nextBtn.addEventListener('click', jumpToNextAnomaly);

// Degradation trend button
const btnTrends = document.getElementById('btn-view-trends');
if (btnTrends) {
  btnTrends.addEventListener('click', () => {
    const card = document.getElementById('trends-card');
    if (card) card.scrollIntoView({ behavior: 'smooth' });
  });
}

// Global Navigation Links
document.querySelectorAll('.global-nav-item').forEach(el => {
  el.addEventListener('click', () => {
    const target = el.getAttribute('data-target');
    if (target && target !== 'physics') {
      selectComponent(target);
    }
  });
});

sizeCanvas();
initThree();
requestAnimationFrame(animate);
initBackend();

/* ============ Backend integration (trained bundle served by FastAPI at /api) ============ */
function showNoBundle(msg) {
  const el = document.getElementById('no-bundle');
  if (el) el.style.display = 'flex';
  const m = document.getElementById('no-bundle-msg');
  if (m) m.textContent = msg;
}
function hideNoBundle() {
  const el = document.getElementById('no-bundle');
  if (el) el.style.display = 'none';
}

function applyPayload(p) {
  backend.payload = p;
  backend.explainKey = null;
  bandEdges = p.band_edges || bandEdges;
  const c = p.columns;
  const g = (k, i) => { const a = c[k]; const v = a ? a[i] : null; return (v === null || v === undefined) ? NaN : v; };
  N = p.n_cycles;
  data = [];
  for (let i = 0; i < N; i++) {
    const hi = {}, kf = {}, kfs = {}, spread = {}, truth = {}, res = {}, x = {}, ideal = {}, nis = {}, gated = {}, theta = {}, thetaSig = {};
    p.components.forEach(cc => {
      hi[cc] = g('hi_' + cc, i); kf[cc] = g('kf_hi_' + cc, i); kfs[cc] = g('kf_hi_' + cc + '_sigma', i);
      spread[cc] = g('hi_' + cc + '_spread', i); truth[cc] = g('truth_hi_' + cc, i);
    });
    p.sensors.forEach(sn => {
      res[sn] = g('res_' + sn, i); x[sn] = g('x_' + sn, i); ideal[sn] = g('ideal_' + sn, i);
      nis[sn] = g('nis_' + sn, i); gated[sn] = !!(c['gated_' + sn] || [])[i];
    });
    (p.theta || []).forEach(t => { theta[t] = g('kf_' + t, i); thetaSig[t] = g('kf_' + t + '_sigma', i); });
    const p30 = isFinite(g('v_P30', i)) ? g('v_P30', i) : x.Ps30;
    const ohiComp = p.components[c.ohi_component ? c.ohi_component[i] : 0];
    data.push({
      cycle: g('cycle', i), alt: g('alt_m', i), mach: g('mach', i), tra: g('tra', i), ttamb: g('t2_K', i), ptamb: x.P2,
      nf: x.Nf, nc: x.Nc, wf: x.Wf, wfIdeal: ideal.Wf, fc: c.fc ? c.fc[i] : null, duration: g('duration_min', i),
      // Brayton stations of the turbofan core: 2 fan inlet -> 30 HPC exit -> 40 burner exit (virtual) -> 50 LPT exit
      pi_c: p30 / x.P2, t2: x.T30, p2: p30, t3: g('v_T40', i), p3: x.P40, t4: x.T50, p4: x.P50, t48: x.T48,
      eta_c: g('p_eta_hpc', i), eta_core: g('p_eta_core', i), eta_t: g('eta_turb_est', i), dp_comb: g('p_burner_dp', i),
      fan_pr: g('p_fan_pr', i), nf_corr: g('p_nf_corr', i),
      hi, kf, kfs, spread, truth, res, x, ideal, nis, gated, theta, thetaSig,
      hi_fan: hi.fan, hi_comp_min: Math.min(hi.lpc, hi.hpc), hi_turb_min: Math.min(hi.hpt, hi.lpt),
      ohi: g('ohi', i), ohiComp, u_ohi: spread[ohiComp],
      hiRaw: p.components.map(cc => g('hi_' + cc + '_raw', i)),
      rul: g('rul', i), q05: g('rul_q05', i), q95: g('rul_q95', i), rulTrees: g('rul_trees_only', i), truthRul: g('truth_rul', i),
    });
  }
}

async function loadEngine(id) {
  if (backend.uploads[id]) {
    applyPayload(backend.uploads[id]);
  } else {
    const r = await fetch('/api/engine/' + encodeURIComponent(id));
    const body = await r.json();
    if (!r.ok) throw new Error(body.detail || r.status);
    applyPayload(body);
  }
  backend.currentId = id;
  selectedEngine = id;
}

async function initBackend() {
  let health;
  try {
    health = await (await fetch('/api/health')).json();
  } catch (e) {
    showNoBundle('Backend not reachable. Start it with: python -m backend');
    return;
  }
  if (!health.bundle_loaded) { showNoBundle(health.detail || 'No model bundle loaded.'); return; }
  const r = await fetch('/api/engines');
  const body = await r.json();
  if (!r.ok) { showNoBundle(body.detail || 'No engines available.'); return; }
  backend.engines = body;
  backend.online = true;
  hideNoBundle();
  engineSelect.innerHTML = backend.engines.map(e =>
    `<option value="${e.engine_id}">${e.dataset || 'Engine'} · unit ${e.unit_local ?? e.engine_id} (N=${e.n_cycles})</option>`).join('');
  const sub = document.getElementById('engine-subtitle');
  if (sub) sub.textContent = `${health.model.dataset} · ${backend.engines.length} test engines`;
  await loadEngine(engineSelect.value);
  showEngine();
}

// Upload CSV -> run the full pipeline on the backend
(function setupUpload() {
  const btn = document.getElementById('upload-csv-btn');
  if (!btn) return;
  const input = document.createElement('input');
  input.type = 'file'; input.accept = '.csv'; input.style.display = 'none';
  document.body.appendChild(input);
  btn.addEventListener('click', () => {
    if (!backend.online) { alert('No model bundle loaded in the backend yet.'); return; }
    input.click();
  });
  input.addEventListener('change', async () => {
    if (!input.files.length) return;
    const fd = new FormData();
    fd.append('file', input.files[0]);
    btn.disabled = true;
    try {
      const r = await fetch('/api/analyze', { method: 'POST', body: fd });
      const body = await r.json();
      if (!r.ok) throw new Error(body.detail || r.status);
      Object.entries(body.engines).forEach(([id, p]) => {
        backend.uploads[id] = p;
        const o = document.createElement('option');
        o.value = id; o.textContent = `Uploaded ${id} (N=${p.n_cycles})`;
        engineSelect.appendChild(o);
      });
      engineSelect.value = Object.keys(body.engines)[0];
      engineSelect.dispatchEvent(new Event('change'));
    } catch (err) {
      alert('Could not analyse CSV: ' + err.message);
    } finally {
      btn.disabled = false; input.value = '';
    }
  });
})();

/* ---- Layer 3-5 panels: model badge, explainability, Kalman filter, alerts ---- */
function renderBackendPanels(i) {
  const p = backend.payload;
  const wrap = document.getElementById('backend-panels');
  if (wrap) wrap.style.display = p ? '' : 'none';
  if (!p) return;
  const d = data[i];

  const badge = document.getElementById('model-badge');
  if (badge) {
    const m = p.model || {};
    const rm = m.metrics && m.metrics.rul_stack;
    badge.textContent = `${m.dataset || ''} · ${m.gru_used ? 'trees + GRU stack' : 'tree stack'}` +
      (rm && isFinite(rm.rmse) ? ` · test RUL RMSE ${rm.rmse.toFixed(1)} cyc` : '');
    badge.style.color = (m.dataset || '').startsWith('STUB') ? '#dc2626' : '#059669';
  }

  const ekf = document.getElementById('ekf-body');
  if (ekf) {
    const pctv = v => isFinite(v) ? (v * 100).toFixed(2) : '—';
    const rows = p.components.map(cc => {
      const comp = cc === 'fan' ? 'fan' : cc.toUpperCase();
      const e = `${comp}_eff_mod`, f = `${comp}_flow_mod`;
      return `<tr class="border-t border-outline-variant/30"><td class="py-1 pr-2 font-bold">${COMP_LABEL[cc]}</td>
        <td class="py-1 pr-2">${pctv(d.theta[e])} <span class="text-outline">±${pctv(d.thetaSig[e])}</span></td>
        <td class="py-1 pr-2">${pctv(d.theta[f])} <span class="text-outline">±${pctv(d.thetaSig[f])}</span></td>
        <td class="py-1" style="color:${bandColorRaw(d.kf[cc])}">${fmtN(d.kf[cc], 2)}</td></tr>`;
    }).join('');
    const sens = p.sensors.map(sn => ({ sn, nis: d.nis[sn], bad: d.gated[sn] }))
      .sort((a, b) => (b.nis || 0) - (a.nis || 0)).slice(0, 4);
    ekf.innerHTML = `<table class="w-full text-[10px] font-data-display">
        <thead><tr class="text-outline text-[9px] uppercase"><th class="text-left">Comp.</th><th class="text-left">&Delta;eff %</th><th class="text-left">&Delta;flow %</th><th class="text-left">HI</th></tr></thead>
        <tbody>${rows}</tbody></table>
      <div class="text-[9px] font-label-sm text-outline uppercase tracking-wider mt-3 mb-1">Largest innovations (sensor check)</div>` +
      sens.map(r => `<div class="flex justify-between text-[11px] font-label-sm"><span class="text-on-surface-variant truncate">${r.sn} · ${p.sensor_info[r.sn]}</span>
        <span class="font-data-display" style="color:${r.bad ? '#dc2626' : '#059669'}">${r.bad ? 'REJECTED' : 'OK'} · NIS ${fmtN(r.nis, 1)}</span></div>`).join('');
  }

  const al = document.getElementById('alerts-body');
  if (al) {
    const active = p.alerts.filter(a => a.cycle <= d.cycle).slice(-6).reverse();
    al.innerHTML = active.length ? active.map(a => `<div class="text-[11px] font-label-sm flex gap-2">
        <span class="font-data-display" style="color:${SEV_COLOR[a.severity] || '#6e7881'}">C${a.cycle}</span>
        <span class="text-on-surface-variant">${a.message}</span></div>`).join('')
      : '<div class="text-[11px] font-label-sm text-on-surface-variant">No threshold breaches so far.</div>';
  }

  requestExplain(i);
}

function renderExplanation(ex, target) {
  const unit = target === 'rul' ? ' cyc' : '';
  const dp = target === 'rul' ? 1 : 3;
  const bar = (v, max) => `<div class="h-1.5 rounded bg-surface-container"><div class="h-full rounded" style="width:${Math.min(100, Math.abs(v) / max * 100).toFixed(0)}%;background:${v < 0 ? '#dc2626' : '#059669'}"></div></div>`;
  const occ = ex.occlusion.sensors.slice(0, 6);
  const omax = Math.max(1e-9, ...occ.map(r => Math.abs(r.value)));
  let html = `<div class="text-[11px] font-label-sm mb-2">Prediction: <b>${ex.prediction.toFixed(dp)}${unit}</b>
    <span class="text-on-surface-variant">(red = pushes ${target === 'rul' ? 'RUL' : 'health'} down)</span></div>
    <div class="text-[10px] font-label-sm text-on-surface-variant mb-1">Full stack incl. GRU &amp; Kalman filter — occlusion per sensor</div>`;
  html += occ.map(r => `<div class="mb-1.5"><div class="flex justify-between text-[11px] font-label-sm">
      <span class="truncate text-on-surface-variant" title="${r.description}">${r.sensor} · ${r.description}</span>
      <span class="font-data-display shrink-0 ml-2 whitespace-nowrap">${r.value >= 0 ? '+' : ''}${r.value.toFixed(dp)}${unit}</span></div>${bar(r.value, omax)}</div>`).join('');
  if (ex.tree_shap) {
    const gs = ex.tree_shap.groups.slice(0, 5);
    const gmax = Math.max(1e-9, ...gs.map(r => Math.abs(r.value)));
    html += `<div class="text-[10px] font-label-sm text-on-surface-variant mt-3 mb-1">Tree models — TreeSHAP by feature group</div>` +
      gs.map(r => `<div class="mb-1.5"><div class="flex justify-between text-[11px] font-label-sm"><span class="truncate text-on-surface-variant">${r.name}</span>
        <span class="font-data-display shrink-0 ml-2 whitespace-nowrap">${r.value >= 0 ? '+' : ''}${r.value.toFixed(dp)}${unit}</span></div>${bar(r.value, gmax)}</div>`).join('');
  } else {
    html += '<div class="text-[10px] font-label-sm text-outline mt-2">TreeSHAP unavailable for this bundle.</div>';
  }
  return html;
}

function requestExplain(i) {
  const box = document.getElementById('explain-body');
  if (!box || !backend.currentId || !data.length) return;
  const tsel = document.getElementById('explain-target');
  const target = tsel ? tsel.value : 'rul';
  const cyc = data[i].cycle;
  const key = `${backend.currentId}:${cyc}:${target}`;
  if (backend.explainKey === key) return;
  clearTimeout(backend.explainTimer);
  backend.explainTimer = setTimeout(async () => {
    box.innerHTML = '<div class="text-[11px] font-label-sm text-on-surface-variant">Computing explanation…</div>';
    try {
      const r = await fetch(`/api/engine/${encodeURIComponent(backend.currentId)}/explain/${cyc}?target=${target}`);
      if (!r.ok) throw new Error(r.status);
      const ex = await r.json();
      if (!data.length || data[currentCycle].cycle !== cyc) return;   // user moved on
      backend.explainKey = key;
      box.innerHTML = renderExplanation(ex, target);
    } catch (e) {
      box.innerHTML = '<div class="text-[11px] font-label-sm text-on-surface-variant">Explanation unavailable.</div>';
    }
  }, 300);
}

const explainSel = document.getElementById('explain-target');
if (explainSel) explainSel.addEventListener('change', () => { backend.explainKey = null; if (data.length) requestExplain(currentCycle); });
