
    // --- ADVANCED ANALYTICS RENDER LOGIC ---

    function renderAdvancedAnalytics(d, ideal, actual, i) {
      // Key Performance Parameters
      const thermEff = actual.eta_th * 100;
      const compEff = d.eta_c * 100;
      const turbEff = d.eta_t * 100;
      const bwr = actual.bwr;
      const cpl = d.dp_comb * 100;

      const f_therm = val => `<span style="color:${val>=45?'var(--teal)':'var(--orange)'}">${val.toFixed(1)}%</span>`;
      const f_comp = val => `<span style="color:${val>=80?'var(--teal)':'var(--orange)'}">${val.toFixed(1)}%</span>`;
      
      const el_therm = document.getElementById('kpi-therm');
      if(el_therm) {
        el_therm.innerHTML = f_therm(thermEff);
        document.getElementById('kpi-comp').innerHTML = f_comp(compEff);
        document.getElementById('kpi-turb').innerHTML = f_comp(turbEff);
        document.getElementById('kpi-bwr').textContent = bwr.toFixed(3);
        document.getElementById('kpi-cpl').innerHTML = `<span style="color:${cpl<=4?'var(--teal)':'var(--orange)'}">${cpl.toFixed(1)}%</span>`;

        // Physics & Model Consistency
        const s_gen = actual.stations[3].s - actual.stations[0].s; 
        document.getElementById('phy-ent').textContent = (s_gen/50).toFixed(1) + ' %';
        document.getElementById('phy-conf').textContent = Math.min(99.8, 90 + d.ohi*9.8).toFixed(1) + ' %';

        // Station Deviation Table
        const stations = ['Inlet (Total T)', 'Compressor Exit (T)', 'Combustor Exit (T)', 'Turbine Exit (T)'];
        let tbHtml = '';
        for (let j=0; j<4; j++){
          const devT = actual.stations[j].T - ideal.stations[j].T;
          let sev = 'Normal';
          let scol = 'var(--teal)';
          if(Math.abs(devT) > 100) { sev='High'; scol='var(--red)'; }
          else if(Math.abs(devT) > 40) { sev='Moderate'; scol='var(--amber)'; }
          
          tbHtml += `<tr>
            <td style="padding:6px 0;">${j+1}</td>
            <td>${stations[j]}</td>
            <td style="text-align:right; font-family:'IBM Plex Mono'; color:${devT>0?'var(--orange)':devT<0?'var(--teal)':'inherit'};">${devT>0?'+':''}${Math.round(devT)} K</td>
            <td style="text-align:right; display:flex; align-items:center; justify-content:flex-end; gap:4px;">
              <div style="width:6px;height:6px;border-radius:50%;background:${scol}"></div>${sev}
            </td>
          </tr>`;
        }
        document.getElementById('stationDevTable').innerHTML = tbHtml;

        // Most Probable Cause Donut Chart
        const compDeg = 1 - d.hi_c;
        const turbDeg = 1 - d.hi_t;
        const sensDeg = 0.05 + Math.random()*0.02; 
        const total = compDeg + turbDeg + sensDeg + 0.001; 
        
        const pComp = (compDeg / total) * 100;
        const pTurb = (turbDeg / total) * 100;
        const pSens = (sensDeg / total) * 100;

        const donut = document.getElementById('causeDonut');
        const r = 36; const c = 50; const circ = 2 * Math.PI * r;
        
        const dash_comp = `${(pComp/100)*circ} ${circ}`;
        const dash_turb = `${(pTurb/100)*circ} ${circ}`;
        const dash_sens = `${(pSens/100)*circ} ${circ}`;
        
        const c_comp = 'var(--cyan)';
        const c_turb = 'var(--teal)';
        const c_sens = 'var(--orange)';

        donut.innerHTML = `
          <circle cx="${c}" cy="${c}" r="${r}" fill="none" stroke="rgba(0,0,0,0.05)" stroke-width="16" />
          <circle cx="${c}" cy="${c}" r="${r}" fill="none" stroke="${c_comp}" stroke-width="16" stroke-dasharray="${dash_comp}" stroke-dashoffset="${0}" transform="rotate(-90 50 50)"/>
          <circle cx="${c}" cy="${c}" r="${r}" fill="none" stroke="${c_turb}" stroke-width="16" stroke-dasharray="${dash_turb}" stroke-dashoffset="${-(pComp/100)*circ}" transform="rotate(-90 50 50)"/>
          <circle cx="${c}" cy="${c}" r="${r}" fill="none" stroke="${c_sens}" stroke-width="16" stroke-dasharray="${dash_sens}" stroke-dashoffset="${-((pComp+pTurb)/100)*circ}" transform="rotate(-90 50 50)"/>
          <text x="${c}" y="${c}" text-anchor="middle" dominant-baseline="middle" font-size="22" font-weight="700" fill="var(--ink)">${Math.round(pComp)}%</text>
        `;

        document.getElementById('causeLegend').innerHTML = `
          <div style="display:flex; justify-content:space-between;">
            <div style="display:flex; align-items:center; gap:6px;"><div style="width:8px;height:8px;background:${c_comp};border-radius:2px;"></div>Compressor Fouling</div>
            <b style="font-family:'IBM Plex Mono'; font-size:12px;">${pComp.toFixed(1)}%</b>
          </div>
          <div style="display:flex; justify-content:space-between;">
            <div style="display:flex; align-items:center; gap:6px;"><div style="width:8px;height:8px;background:${c_turb};border-radius:2px;"></div>Turbine Erosion</div>
            <b style="font-family:'IBM Plex Mono'; font-size:12px;">${pTurb.toFixed(1)}%</b>
          </div>
          <div style="display:flex; justify-content:space-between;">
            <div style="display:flex; align-items:center; gap:6px;"><div style="width:8px;height:8px;background:${c_sens};border-radius:2px;"></div>Sensor Bias</div>
            <b style="font-family:'IBM Plex Mono'; font-size:12px;">${pSens.toFixed(1)}%</b>
          </div>
        `;

        drawTrendChart(i);
      }
    }

    function drawTrendChart(upTo) {
      const svg = document.getElementById('trendLinesChart');
      if(!svg) return;
      const rect = svg.getBoundingClientRect();
      const w = rect.width || 600, h = 160;
      const padL = 30, padR = 20, padT = 10, padB = 20;
      const iw = w - padL - padR, ih = h - padT - padB;
      
      const slice = data.slice(0, upTo+1);
      
      const d0 = data[0];
      const getDev = (v, v0) => ((v - v0) / v0) * 100;
      
      const ptsComp = slice.map((d, i) => ({ x: padL + (i/(N-1))*iw, y: getDev(d.eta_c, d0.eta_c) }));
      const ptsTurb = slice.map((d, i) => ({ x: padL + (i/(N-1))*iw, y: getDev(d.eta_t, d0.eta_t) }));
      const ptsSFC = slice.map((d, i) => ({ x: padL + (i/(N-1))*iw, y: getDev(d.wf/d.thrust, d0.wf/d0.thrust) }));
      const ptsThrust = slice.map((d, i) => ({ x: padL + (i/(N-1))*iw, y: getDev(d.thrust, d0.thrust) }));

      const yMax = 10;
      const yMin = -15;
      
      const scaleY = (y) => padT + (1 - (Math.max(yMin, Math.min(yMax, y)) - yMin)/(yMax - yMin)) * ih;

      let html = '';
      
      for(let val of [10, 5, 0, -5, -10]){
        const yy = scaleY(val);
        html += `<line x1="${padL}" y1="${yy}" x2="${w-padR}" y2="${yy}" stroke="rgba(0,0,0,0.05)" stroke-width="1"/>`;
        html += `<text x="${padL-4}" y="${yy+3}" text-anchor="end" font-size="9" fill="var(--ink-faint)" font-family="IBM Plex Mono">${val>0?'+':''}${val}</text>`;
      }
      
      const makeLine = (pts, col) => {
        let p = pts.map(pt => `${pt.x},${scaleY(pt.y)}`).join(' ');
        return `<polyline points="${p}" fill="none" stroke="${col}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>`;
      };
      
      html += makeLine(ptsComp, 'var(--cyan)'); 
      html += makeLine(ptsTurb, 'var(--teal)'); 
      html += makeLine(ptsSFC, 'var(--orange)');  
      html += makeLine(ptsThrust, '#a874c7'); 
      
      const cx = padL + (upTo/(N-1))*iw;
      html += `<line x1="${cx}" y1="${padT}" x2="${cx}" y2="${h-padB}" stroke="rgba(0,0,0,0.2)" stroke-dasharray="2,2"/>`;
      
      for(let cyc=0; cyc<=200; cyc+=50){
         const xx = padL + (cyc/(N-1))*iw;
         html += `<text x="${xx}" y="${h-4}" text-anchor="middle" font-size="9" fill="var(--ink-faint)" font-family="IBM Plex Mono">${cyc}</text>`;
      }

      svg.innerHTML = html;
    }
    
